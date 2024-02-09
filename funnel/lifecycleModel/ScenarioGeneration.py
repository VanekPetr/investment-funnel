import numpy as np
import pandas as pd
from loguru import logger
from typing import Tuple
from arch import arch_model

class MomentGenerator(object):
    """
    Provides methods for mean, variace generation.
    """

    @staticmethod
    def _alpha_numerator(Z, S):
        s = 0
        T = Z.shape[1]
        for k in range(T):
            z = Z[:, k][:, np.newaxis]
            X = z @ z.T - S
            s += np.trace(X @ X)
        s /= (T ** 2)
        return s

    @staticmethod
    def _ledoit_wolf_shrinkage(X, S):
        """
        Computes the Ledoit--Wolf shrinkage, using a target of scaled identity.
        """
        N = len(X.columns)
        # In case only one asset in the matrix, for example for benchmark with one asset, no shrinkage is needed
        if N == 1:
            return S

        # Center the data
        X = (X - X.mean(0)).to_numpy().T

        # Target.
        s_avg2 = np.trace(S) / N
        B = s_avg2 * np.eye(N)

        # Shrinkage coefficient.
        alpha_num = MomentGenerator._alpha_numerator(X, S)
        alpha_den = np.trace((S - B) @ (S - B))
        alpha = alpha_num / alpha_den

        # Shrunk covariance
        shrunk = (1 - alpha) * S + alpha * B

        return shrunk

    @staticmethod
    def generate_annual_sigma_mu_with_risk_free(data: pd.DataFrame, risk_free_rate_annual: float = 0.02) -> Tuple[
        pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Computes the annualized and weekly covariance matrix (sigma) and mean return array (mu)
        for the entire historical dataset, including a risk-free asset.

        Parameters:
        - data: A pandas DataFrame with weekly returns for each asset.
        - risk_free_rate_annual: Annual return rate of the risk-free asset, default is 2%.

        Returns:
        - A tuple containing:
            - sigma_annual: Annualized covariance matrix including the risk-free asset.
            - mu_annual: Annualized mean return vector including the risk-free asset.
            - sigma_weekly: Weekly covariance matrix including the risk-free asset.
            - mu_weekly: Weekly mean return vector including the risk-free asset.
        """
        logger.debug(f'Generating annual Sigma and Mu parameter estimations for the optimization model.')
        # Compute the sample covariance matrix for the entire dataset
        sigma_weekly_np = np.atleast_2d(np.cov(data, rowvar=False, bias=True))  # The sample covariance matrix

        # Add a shrinkage term (Ledoit--Wolf multiple of identity)
        sigma_weekly_np = MomentGenerator._ledoit_wolf_shrinkage(data, sigma_weekly_np)

        # Compute the mean return array for the entire dataset
        mu_weekly_np = np.mean(data, axis=0)

        # Convert the annual risk-free rate to a weekly rate
        risk_free_rate_weekly = (1 + risk_free_rate_annual) ** (1 / 52) - 1

        # Append the risk-free rate to the weekly mean return array
        mu_weekly_np = np.append(mu_weekly_np, risk_free_rate_weekly)

        # Append a row and column of zeros for the risk-free asset in the covariance matrix
        sigma_weekly_np = np.pad(sigma_weekly_np, ((0, 1), (0, 1)), 'constant')

        # Convert numpy arrays to pandas DataFrame/Series and set appropriate asset names
        assets_with_rf = data.columns.tolist() + ['Cash']
        sigma_weekly = pd.DataFrame(sigma_weekly_np, index=assets_with_rf, columns=assets_with_rf)
        mu_weekly = pd.Series(mu_weekly_np, index=assets_with_rf)

        # Annualize the covariance matrix and mean return array
        sigma_annual = sigma_weekly * 52
        mu_annual = mu_weekly.copy()
        mu_annual.iloc[:-1] = mu_annual.iloc[:-1] * 52  # Annualize only the risky assets

        return sigma_annual, mu_annual, sigma_weekly, mu_weekly

class ScenarioGenerator(object):
    """
    Provides methods for scenario generation.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def MC_simulation_annual_from_weekly(self, weekly_mu: pd.Series, weekly_sigma: pd.DataFrame, n_simulations: int, n_years: int, cash_return_annual: float = 0.02):
        """
        Generates Monte Carlo simulations for annual returns based on provided weekly mu and sigma.
        Assumes 'Cash' or risk-free asset is already included and sets its annual return to a constant value.

        Parameters:
        - weekly_mu: Weekly mean returns as a pandas Series, including 'Cash'.
        - weekly_sigma: Weekly covariance matrix as a pandas DataFrame, including 'Cash'.
        - n_simulations: Number of simulations to generate.
        - n_years: Number of years to simulate.
        - cash_return_annual: Annual return rate of the 'Cash' or risk-free asset, default is 2%.

        Returns:
        - annual_simulations: An array of simulated annual returns (n_simulations, n_years, n_assets).
        """
        logger.debug(
            f"Simulating annual returns with Monte Carlo method based on weekly mu and weekly sigma. "
            f"We are generating {n_simulations} simulations for {n_years} years.")
        n_assets = len(weekly_mu)
        weeks_per_year = 52
        weekly_scenarios = np.zeros((n_simulations, n_years * weeks_per_year, n_assets), dtype=float)

        # Generate weekly simulations
        for week in range(n_years * weeks_per_year):
            weekly_returns = self.rng.multivariate_normal(mean=weekly_mu.values, cov=weekly_sigma.values, size=n_simulations)
            weekly_scenarios[:, week, :] = weekly_returns

        # Convert weekly simulations to annual simulations
        annual_simulations = np.zeros((n_simulations, n_years, n_assets), dtype=float)
        for year in range(n_years):
            start_week = year * weeks_per_year
            end_week = (year + 1) * weeks_per_year
            # Accumulate weekly returns to get annual returns
            for simulation in range(n_simulations):
                # Convert weekly returns to cumulative product for each asset
                for asset in range(n_assets):
                    if weekly_mu.index[asset] == "Cash":  # Assume 'Cash' represents the risk-free asset
                        # Set 'Cash' returns to a constant annual rate
                        annual_simulations[simulation, year, asset] = cash_return_annual
                    else:
                        annual_simulations[simulation, year, asset] = np.prod(1 + weekly_scenarios[simulation, start_week:end_week, asset]) - 1

        return annual_simulations


    def FHS(self, data: pd.DataFrame, n_simulations: int, n_years: int):
        '''
        This code snippet directly simulates annual returns from the adjusted weekly returns,
        reflecting the current market volatility for each asset and bypassing the need for explicit weekly simulations.
        The process involves:

        1.  Estimating the volatility for each asset using a GARCH model.
        2.  Standardizing the weekly returns by this estimated volatility.
        3.  Adjusting these standardized returns by the most recent volatility estimate to reflect current market conditions.
        4.  For each scenario and year, randomly drawing 52 adjusted weekly returns to represent a year,
            aggregating these to simulate an annual return for each asset.
        '''

        n_assets = len(data.columns) + 1  # Add one for the risk-free asset

        # Initialize the ndarray for annual returns including the risk-free asset
        annual_returns = np.zeros((n_simulations, n_years, n_assets))

        rng = np.random.default_rng()

        # Process each asset (excluding the risk-free asset)
        for asset_idx, asset in enumerate(data.columns):
            # Fit GARCH model to estimate volatility
            garch_model = arch_model(data[asset], p=1, q=1)
            model_result = garch_model.fit(disp='off')  # Suppress optimizer output
            volatility = model_result.conditional_volatility
            standardized_returns = data[asset] / volatility
            current_volatility = volatility.iloc[-1]
            adjusted_returns = standardized_returns * current_volatility

            # Generate annual returns directly from adjusted returns for each scenario
            for scenario in range(n_simulations):
                for year in range(n_years):
                    # Randomly draw 52 weekly returns to simulate a year, then aggregate to an annual return
                    weekly_draws = rng.choice(adjusted_returns, size=52)
                    annual_return = np.prod(1 + weekly_draws) - 1
                    annual_returns[scenario, year, asset_idx] = annual_return

        # Set the 20th asset (risk-free asset) with a constant return of 0.02 for all scenarios and years
        annual_returns[:, :, -1] = 0.02
        return annual_returns
