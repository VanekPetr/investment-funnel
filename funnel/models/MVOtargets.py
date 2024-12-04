from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .ScenarioGeneration import MomentGenerator


# FUNCTION RUNNING THE OPTIMIZATION
# ----------------------------------------------------------------------
def portfolio_risk_target(covariance: np.array) -> float:
    # Fixed equal weight x
    n = covariance.shape[0]
    x = np.ones(n) / n

    # Volatility
    portfolio_vty = np.sqrt(x @ covariance @ x)

    return portfolio_vty


# ----------------------------------------------------------------------
# Mathematical Optimization: TARGETS GENERATION
# ----------------------------------------------------------------------
def get_mvo_targets(
    test_date: str, benchmark: list, budget: int, data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"🎯 Generating Volatility targets for {benchmark}")

    # Define Benchmark
    tickers = benchmark
    # Get weekly return of our benchmark
    whole_dataset_benchmark = data[tickers].copy()

    # Get weekly data just for testing period
    test_dataset_benchmark = whole_dataset_benchmark[
        whole_dataset_benchmark.index >= test_date
    ]

    # Number of weeks for testing
    weeks_n = len(test_dataset_benchmark.index)

    # Get parameters
    sigma_lst, _ = MomentGenerator.generate_sigma_mu_for_test_periods(
        whole_dataset_benchmark, weeks_n
    )

    # Compute the optimal portfolio outperforming zero percentage return
    # ----------------------------------------------------------------------
    p_points = len(sigma_lst)  # number of periods

    # COMPUTE MVO TARGETS
    list_targets = []
    for p in range(p_points):
        # Get parameters for a given period p
        sigma = sigma_lst[p]

        # Compute volatility targets
        vty_target = portfolio_risk_target(sigma)

        # save the result
        list_targets.append(vty_target)

    # Generate new column so that dtype is set right.
    targets = pd.DataFrame(columns=["Vty_Target"], data=list_targets)

    # COMPUTE PORTFOLIO VALUE
    list_portfolio_values = []
    for w in test_dataset_benchmark.index:
        budget_next = sum(
            (budget / len(tickers)) * (1 + test_dataset_benchmark.loc[w, :])
        )
        list_portfolio_values.append(budget_next)
        budget = budget_next

    # Generate dataframe so that dtype is set right.
    portfolio_value = pd.DataFrame(
        columns=["Benchmark_Value"],
        index=test_dataset_benchmark.index,
        data=list_portfolio_values,
    )

    return targets, portfolio_value
