import numpy as np
import math
import pandas as pd
from loguru import logger


class ScenarioGenerator(object):
    """
    Provides methods for scenario generation.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    @staticmethod
    def generate_sigma_mu_for_test_periods(data: pd.DataFrame, n_test: int) -> (list, list):
        logger.debug(f"Computing covariance matrix and mean array for each investment period")

        # Initialize variables
        sigma_lst = []
        mu_lst = []

        n_iter = 4  # we work with 4-week periods
        n_train_weeks = len(data.index) - n_test
        n_rolls = math.floor(n_test / n_iter) + 1

        for p in range(int(n_rolls)):
            rolling_train_dataset = data.iloc[(n_iter * p): (n_train_weeks + n_iter * p), :]

            sigma = np.cov(rolling_train_dataset, rowvar=False)  # The covariance matrix
            # RHO = np.corrcoef(ret_train, rowvar=False)            # The correlation matrix
            mu = np.mean(rolling_train_dataset, axis=0)  # The mean array
            # sd = np.sqrt(np.diagonal(SIGMA))                      # The standard deviation

            sigma_lst.append(sigma)
            mu_lst.append(mu)

        return sigma_lst, mu_lst

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
