from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .ScenarioGeneration import ScenarioGenerator


# Primal CVaR formula
def CVaR(alpha: float, p: np.array, q: np.array) -> Tuple[float, float]:
    """
    Computes CVaR using primal formula.
    NOTE: Inputs p and q should be numpy arrays.
    """
    # We need to be careful that math index starts from 1 but numpy starts from 0
    # (matters in formulas like ceil(alpha * T))
    # T = q.shape[0]
    sort_idx = np.argsort(q)
    sorted_q = q[sort_idx]
    sorted_p = p[sort_idx]

    # Starting index
    i_alpha = np.sort(np.nonzero(np.cumsum(sorted_p) >= alpha)[0])[0]

    # Weight of VaR component in CVaR
    lambda_alpha = (np.sum(sorted_p[: (i_alpha + 1)]) - alpha) / (1 - alpha)

    # CVaR
    var = sorted_q[i_alpha]
    cvar = lambda_alpha * sorted_q[i_alpha] + np.dot(
        sorted_p[(i_alpha + 1) :], sorted_q[(i_alpha + 1) :]
    ) / (1 - alpha)

    return var, cvar


# FUNCTION RUNNING THE OPTIMIZATION
# ----------------------------------------------------------------------
def portfolio_risk_target(scenarios: pd.DataFrame, cvar_alpha: float) -> float:
    # Fixed equal weight x
    x = pd.Series(index=scenarios.columns, data=1 / scenarios.shape[1])

    # Number of scenarios
    scenario_n = scenarios.shape[0]

    # Portfolio loss scenarios
    losses = (-scenarios @ x).to_numpy()

    # Probabilities
    probs = np.ones(scenario_n) / scenario_n

    # CVaR
    _, portfolio_cvar = CVaR(1 - cvar_alpha, probs, losses)

    return portfolio_cvar


# ----------------------------------------------------------------------
# Mathematical Optimization: TARGETS GENERATION
# ----------------------------------------------------------------------
def get_cvar_targets(
    test_date: str,
    benchmark: list,
    budget: int,
    cvar_alpha: float,
    data: pd.DataFrame,
    scgen: ScenarioGenerator,
    n_simulations: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"🎯 Generating CVaR targets for {benchmark}")

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

    # Get scenarios
    # The Monte Carlo Method
    target_scenarios = scgen.bootstrapping(
        data=whole_dataset_benchmark,  # subsetMST or subsetCLUST
        n_simulations=n_simulations,
        n_test=weeks_n,
    )

    # Compute the optimal portfolio outperforming zero percentage return
    # ----------------------------------------------------------------------
    p_points = len(target_scenarios[:, 0, 0])  # number of periods
    s_points = len(target_scenarios[0, :, 0])  # number of scenarios

    # COMPUTE CVaR TARGETS
    list_targets = []
    for p in range(p_points):
        # create data frame with scenarios for a given period p
        scenario_df = pd.DataFrame(
            target_scenarios[p, :, :], columns=tickers, index=list(range(s_points))
        )

        # run CVaR model to compute CVaR targets
        cvar_target = portfolio_risk_target(
            scenarios=scenario_df, cvar_alpha=cvar_alpha
        )
        # save the result
        list_targets.append(cvar_target)

    # Generate new column so that dtype is set right.
    targets = pd.DataFrame(columns=["CVaR_Target"], data=list_targets)

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
