import numpy as np
import pandas as pd
from loguru import logger


# Primal CVaR formula
def CVaR(alpha, p, q):
    """
    Computes CVaR using primal formula. 
    NOTE: Inputs p and q should be numpy arrays.
    """
    # We need to be careful that math index starts from 1 but numpy starts from 0 (matters in formulas like ceil(alpha * T))
    T = q.shape[0]
    sort_idx = np.argsort(q)
    sorted_q = q[sort_idx]
    sorted_p = p[sort_idx]
    
    # Starting index 
    i_alpha = np.sort(np.nonzero(np.cumsum(sorted_p) >= alpha)[0])[0]

    # Weight of VaR component in CVaR
    lambda_alpha = (np.sum(sorted_p[:(i_alpha + 1)]) - alpha) / (1 - alpha)    
    
    # CVaR
    var = sorted_q[i_alpha]
    cvar = lambda_alpha * sorted_q[i_alpha] + np.dot(sorted_p[(i_alpha + 1):], sorted_q[(i_alpha + 1):]) / (1 - alpha)
    
    return var, cvar


# FUNCTION RUNNING THE OPTIMIZATION
# ----------------------------------------------------------------------
def portfolio_risk_target(scenarios, cvar_alpha):
    
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
def get_risk_targets(test_date, benchmark, budget, data, scgen):

    # Define Benchmark
    tickers = benchmark

    # *** For Morningstar data ***
    target_weekly_ret = data[tickers].copy()

    # Get weekly data for testing
    test_weekly_ret = target_weekly_ret[target_weekly_ret.index >= test_date]

    # Number of weeks for testing
    weeks_n = len(test_weekly_ret.index)

    # Get scenarios
    # The bootstrapping method
    target_scenarios = scgen.bootstrapping(
        data=target_weekly_ret,       # subsetMST or subsetCLUST
        n_simulations=250,
        n_test=weeks_n
    )

    # Compute the optimal portfolio outperforming zero percentage return
    # ----------------------------------------------------------------------
    p_points = len(target_scenarios[:, 0, 0])       # number of periods
    s_points = len(target_scenarios[0, :, 0])       # number of scenarios

    # COMPUTE CVaR TARGETS
    list_targets = []
    for p in range(p_points):
        # create data frame with scenarios for a given period p
        scenario_df = pd.DataFrame(target_scenarios[p, :, :],
                                   columns=tickers,
                                   index=list(range(s_points)))

        # run CVaR model to compute CVaR targets
        cvar_target = portfolio_risk_target(scenarios=scenario_df,
                                            cvar_alpha=cvar_alpha)
        # save the result
        list_targets.append(cvar_target)
    
    # Generate new column so that dtype is set right.
    targets = pd.DataFrame(columns=["CVaR_Target"], data=list_targets)

    # COMPUTE PORTFOLIO VALUE
    list_portfolio_values = []
    for w in test_weekly_ret.index:
        budget_next = sum((budget/len(tickers)) * (1 + test_weekly_ret.loc[w, :])) 
        list_portfolio_values.append(budget_next)
        budget = budget_next

    # Generate dataframe so that dtype is set right.
    portfolio_value = pd.DataFrame(columns=["Benchmark_Value"], index=test_weekly_ret.index, data=list_portfolio_values)

    return targets, portfolio_value
