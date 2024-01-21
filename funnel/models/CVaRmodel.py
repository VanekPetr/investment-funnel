import pickle
from typing import Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
from loguru import logger


# ----------------------------------------------------------------------
# MODEL FOR OPTIMIZING THE BACKTEST PERIODS
# ----------------------------------------------------------------------
def rebalancing_model(
    mu,
    scenarios,
    cvar_targets,
    cvar_alpha,
    cash,
    x_old,
    trans_cost,
    max_weight,
    solver,
    inaccurate,
    lower_bound,
):
    """This function finds the optimal enhanced index portfolio according to some benchmark.
    The portfolio corresponds to the tangency portfolio where risk is evaluated according to
    the CVaR of the tracking error. The model is formulated using fractional programming.

    Parameters
    ----------
    mu : pandas.Series with float values
        asset point forecast
    scenarios : pandas.DataFrame with float values
        Asset scenarios
    cvar_targets:
        cvar targets for our optimal portfolio
    cash:
        additional budget for our portfolio
    x_old:
        old portfolio allocation
    trans_cost:
        transaction costs
    max_weight : float
        Maximum allowed weight
    cvar_alpha : float
        Alpha value used to evaluate Value-at-Risk one
    solver: str
        The name of the solver to use, as returned by cvxpy.installed_solvers()
    inaccurate: bool
        Whether to also use solution with status "optimal_inaccurate"
    lower_bound: int
        Minimum weight given to each selected asset.

    Returns
    -------
    float
        Asset weights in an optimal portfolio
    """

    # Define index
    i_idx = scenarios.columns
    N = i_idx.size

    # Number of scenarios
    T = scenarios.shape[0]
    # Variable transaction costs
    c = trans_cost

    # Define variables
    # - portfolio
    x = cp.Variable(N, name="x", nonneg=True)
    # - |x - x_old|
    absdiff = cp.Variable(N, name="absdiff", nonneg=True)
    # - cost
    cost = cp.Variable(name="cost", nonneg=True)
    # - loss deviation
    vardev = cp.Variable(T, name="vardev", nonneg=True)
    # - VaR and CVaR
    var = cp.Variable(name="var", nonneg=True)
    cvar = cp.Variable(name="cvar", nonneg=True)

    # Define objective (max expected portfolio return)
    objective = cp.Maximize(mu.to_numpy() @ x)

    # Define constraints
    constraints = [
        # - VaR deviation
        -scenarios.to_numpy() @ x - var <= vardev,
        # - CVaR limit
        var + 1 / (T * cvar_alpha) * cp.sum(vardev) == cvar,
        cvar <= cvar_targets,
        # - Cost of rebalancing
        c * cp.sum(absdiff) == cost,
        x - x_old <= absdiff,
        x - x_old >= -absdiff,
        # - Budget
        x_old.sum() + cash - cp.sum(x) - cost == 0,
        # - Concentration limits
        x <= max_weight * cp.sum(x),
    ]

    if lower_bound != 0:
        z = cp.Variable(
            N, boolean=True
        )  # Binary variable indicates if asset is selected
        upper_bound = 100

        constraints.append(lower_bound * z <= x)
        constraints.append(x <= upper_bound * z)
        constraints.append(cp.sum(z) >= 1)

    # Define model
    model = cp.Problem(objective=objective, constraints=constraints)

    # Solve
    model.solve(solver=solver, verbose=False)

    # Get positions
    accepted_statuses = ["optimal"]
    if inaccurate:
        accepted_statuses.append("optimal_inaccurate")
    if model.status in accepted_statuses:
        opt_port = pd.Series(x.value, index=mu.index)

        # Set floating data points to zero and normalize
        opt_port[np.abs(opt_port) < 0.000001] = 0
        port_val = np.sum(opt_port)
        cvar_result_p = cvar.value / port_val
        opt_port = opt_port / port_val

        # Remaining cash
        cash = cash - (port_val + cost.value - x_old.sum())

        # return portfolio, CVaR, and alpha
        return opt_port, cvar_result_p, port_val, cash

    else:
        # Save inputs, so that failing problem can be investigated separately e. g. in a notebook
        inputs = {
            "mu": mu,
            "scenarios": scenarios,
            "cvar_targets": cvar_targets,
            "cvar_alpha": cvar_alpha,
            "cash": cash,
            "x_old": x_old,
            "trans_cost": trans_cost,
            "max_weight": max_weight,
        }
        file = open("rebalance_inputs.pkl", "wb")
        pickle.dump(inputs, file)
        file.close()

        # Print an error if the model is not optimal
        logger.exception(
            f"❌ Solver does not find optimal solution. Status code is {model.status}"
        )


# ----------------------------------------------------------------------
# Mathematical Optimization: RUN THE CVAR MODEL
# ----------------------------------------------------------------------
def cvar_model(
    test_ret: pd.DataFrame,
    scenarios: np.ndarray,
    targets: pd.DataFrame,
    budget: float,
    cvar_alpha: float,
    trans_cost: float,
    max_weight: float,
    solver: str,
    inaccurate: bool = True,
    lower_bound=int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Method to run the CVaR model over given periods
    """
    p_points, s_points, _ = scenarios.shape  # number of periods, number of scenarios
    prob = 1 / s_points  # probability of each scenario

    assets = test_ret.columns  # names of all assets

    # LIST TO STORE CVaR TARGETS
    list_portfolio_cvar = []
    # LIST TO STORE VALUE OF THE PORTFOLIO
    list_portfolio_value = []
    # LIST TO STORE PORTFOLIO ALLOCATION
    list_portfolio_allocation = []

    x_old = pd.Series(0, index=assets)
    cash = budget
    portfolio_value_w = budget

    logger.info(f"🤖 Selected solver is {solver}")
    for p in range(p_points):
        logger.info(f"🚀 Optimizing period {p + 1} out of {p_points}.")

        # Create dataframe with scenarios for a period p
        scenarios_df = pd.DataFrame(scenarios[p, :, :], columns=test_ret.columns)

        # compute expected returns of all assets (EP)
        expected_returns = sum(
            prob * scenarios_df.loc[i, :] for i in scenarios_df.index
        )

        # run CVaR model
        p_alloc, cvar_val, port_val, cash = rebalancing_model(
            mu=expected_returns,
            scenarios=scenarios_df,
            cvar_targets=targets.loc[p, "CVaR_Target"] * portfolio_value_w,
            cvar_alpha=cvar_alpha,
            cash=cash,
            x_old=x_old,
            trans_cost=trans_cost,
            max_weight=max_weight,
            solver=solver,
            inaccurate=inaccurate,
            lower_bound=lower_bound,
        )

        # save the result
        list_portfolio_cvar.append(cvar_val)
        # save allocation
        list_portfolio_allocation.append(p_alloc)

        portfolio_value_w = port_val
        # COMPUTE PORTFOLIO VALUE
        for w in test_ret.index[(p * 4) : (4 + p * 4)]:
            portfolio_value_w = sum(
                p_alloc * portfolio_value_w * (1 + test_ret.loc[w, assets])
            )
            list_portfolio_value.append((w, portfolio_value_w))

        x_old = p_alloc * portfolio_value_w

    portfolio_cvar = pd.DataFrame(columns=["CVaR"], data=list_portfolio_cvar)
    portfolio_value = pd.DataFrame(
        columns=["Date", "Portfolio_Value"], data=list_portfolio_value
    ).set_index("Date", drop=True)
    portfolio_allocation = pd.DataFrame(columns=assets, data=list_portfolio_allocation)

    return portfolio_allocation, portfolio_value, portfolio_cvar
