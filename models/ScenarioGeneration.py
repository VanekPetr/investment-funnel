import numpy as np
import math
import pandas as pd
from loguru import logger
from typing import Tuple, List


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
        s /= (T**2)
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
    def generate_sigma_mu_for_test_periods(data: pd.DataFrame, n_test: int) -> Tuple[List, List]:
        logger.debug(f"Computing covariance matrix and mean array for each investment period")

        # Initialize variables
        sigma_lst = []
        mu_lst = []

        n_iter = 4  # we work with 4-week periods
        n_train_weeks = len(data.index) - n_test
        n_rolls = math.floor(n_test / n_iter) + 1

        for p in range(int(n_rolls)):
            rolling_train_dataset = data.iloc[(n_iter * p): (n_train_weeks + n_iter * p), :]

            sigma = np.atleast_2d(np.cov(rolling_train_dataset, rowvar=False, bias=True))     # The sample covariance matrix

            # Add a shrinkage term (Ledoit--Wolf multiple of identity)
            sigma = MomentGenerator._ledoit_wolf_shrinkage(rolling_train_dataset, sigma)

            # Make sure sigma is positive semidefinite
            # sigma = np.atleast_2d(0.5 * (sigma + sigma.T))
            # min_eig = np.min(np.linalg.eigvalsh(sigma))
            # if min_eig < 0:
            #     sigma -= 5 * min_eig * np.eye(*sigma.shape)

            # RHO = np.corrcoef(ret_train, rowvar=False)            # The correlation matrix
            mu = np.mean(rolling_train_dataset, axis=0)             # The mean array
            # sd = np.sqrt(np.diagonal(SIGMA))                      # The standard deviation

            sigma_lst.append(sigma)
            mu_lst.append(mu)

        return sigma_lst, mu_lst


class ScenarioGenerator(object):
    """
    Provides methods for scenario generation.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    # ----------------------------------------------------------------------
    # Scenario Generation: THE MONTE CARLO METHOD
    # ----------------------------------------------------------------------
    def monte_carlo(
            self, data: pd.DataFrame, n_simulations: int, n_test: int, sigma_lst: list, mu_lst: list
    ) -> np.ndarray:
        logger.debug(f"Generating {n_simulations} scenarios for each investment period with Monte Carlo method")

        n_iter = 4  # we work with 4-week periods
        n_indices = data.shape[1]
        n_rolls = math.floor(n_test / n_iter) + 1
        sim = np.zeros((n_rolls*4, n_simulations, n_indices), dtype=float)  # Match GAMS format

        # First generate the weekly simulations for each rolling period
        for p in range(int(n_rolls)):
            sigma = sigma_lst[p]
            mu = mu_lst[p]

            for week in range(n_iter*p, n_iter*p+n_iter):
                sim[week, :, :] = self.rng.multivariate_normal(mean=mu, cov=sigma, size=n_simulations)

        # Now create the monthly (4-weeks) simulations for each rolling period
        monthly_sim = np.zeros((n_rolls, n_simulations, n_indices))
        for roll in range(n_rolls):
            roll_mult = roll * n_iter
            for s in range(n_simulations):
                for index in range(n_indices):
                    tmp_rets = 1 + sim[roll_mult:(roll_mult + n_iter), s, index]
                    monthly_sim[roll, s, index] = np.prod(tmp_rets) - 1

        return monthly_sim

    # ----------------------------------------------------------------------
    # Scenario Generation: THE BOOTSTRAPPING METHOD
    # ----------------------------------------------------------------------
    def bootstrapping(self, data: pd.DataFrame, n_simulations: int, n_test: int) -> np.ndarray:
        logger.debug(f"Generating {n_simulations} scenarios for each investment period with Bootstrapping method")

        n_iter = 4  # 4 weeks compounded in our scenario                                                         
        n_train_weeks = len(data.index) - n_test
        n_indices = data.shape[1]
        n_simulations = n_simulations
        n_rolls = math.floor(n_test / n_iter) + 1

        sim = np.zeros((int(n_rolls), n_simulations, n_indices, n_iter), dtype=float)
        monthly_sim = np.ones((int(n_rolls), n_simulations, n_indices,))
        for p in range(int(n_rolls)):
            for s in range(n_simulations):
                for w in range(n_iter):
                    random_num = self.rng.integers(n_iter * p, n_train_weeks + n_iter * p)
                    sim[p, s, :, w] = data.iloc[random_num, :]
                    monthly_sim[p, s, :] *= (1 + sim[p, s, :, w])
                monthly_sim[p, s, :] += -1

        return monthly_sim
