import cvxpy as cp
import numpy as np
import pandas as pd
import pickle
from typing import Tuple
from loguru import logger


# ----------------------------------------------------------------------
# MODEL FOR OPTIMIZING THE BACKTEST PERIODS 
# ----------------------------------------------------------------------
def rebalancing_model(mu, covariance, vty_targets, cash, x_old, trans_cost, max_weight, solver, inaccurate):
    """ This function finds the optimal enhanced index portfolio according to some benchmark.
    The portfolio corresponds to the tangency portfolio where risk is evaluated according to 
    the volatility of the tracking error. The model is formulated using quadratic programming.
    
    Parameters
    ----------
    mu : pandas.Series with float values
        asset point forecast
    covariance : pandas.DataFrame with covariances
        Asset covariances
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
    solver: str
        The name of the solver to use, as returned by cvxpy.installed_solvers()  
    inaccurate: bool
        Whether to also use solution with status "optimal_inaccurate". 

    Returns
    -------
    float
        Asset weights in an optimal portfolio 
    """ 
    # Number of assets
    N = covariance.columns.size
    
    # Variable transaction costs
    c = trans_cost

    # Factorize the covariance
    G = np.linalg.cholesky(covariance)

    # Define variables
    # - portfolio
    x = cp.Variable(N, name="x", nonneg=True)
    # - |x - x_old|
    absdiff = cp.Variable(N, name="absdiff", nonneg=True) 
    # - volatility
    vty = cp.Variable(name="vty", nonneg=True)    
    # - cost
    cost = cp.Variable(name="cost", nonneg=True) 

    # Define objective (max expected portfolio return)
    objective = cp.Maximize(mu.to_numpy() @ x)

    # Define constraints
    constraints = [
        # - Volatility limit
        cp.norm(G @ x) == vty
        vty <= vty_target

        # - Cost of rebalancing
        c * cp.sum(absdiff) == cost,
        x - x_old <= absdiff,
        x - x_old >= -absdiff,

        # - Budget
        x_old.sum() + cash - cp.sum(x) - cost == 0,

        # - Concentration limits
        x <= max_weight * cp.sum(x)
    ]

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
        vty_result_p = vty.value / port_val
        opt_port = opt_port / port_val

        # Remaining cash
        cash = cash - (port_val + cost.value - x_old.sum())
      
        # return portfolio, CVaR, and alpha
        return opt_port, vty_result_p, port_val, cash
             
    else:
        # Save inputs, so that failing problem can be investigated separately e. g. in a notebook
        inputs = {
             "mu": mu, 
             "covariance": covariance, 
             "vty_targets": vty_targets, 
             "cash": cash, 
             "x_old": x_old, 
             "trans_cost": trans_cost, 
             "max_weight": max_weight
        }
        file = open('rebalance_inputs.pkl', 'wb')
        pickle.dump(inputs, file)
        file.close()
        
        # Print an error if the model is not optimal
        logger.exception(f"Solver does not find optimal solution. Status code is {model.status}")
 

# ----------------------------------------------------------------------
# Mathematical Optimization: RUN THE CVAR MODEL
# ----------------------------------------------------------------------
def volatility_model(
        test_ret: pd.DataFrame,
        mu_lst: list,
        sigma_lst: list,
        targets: pd.DataFrame,
        budget: float,
        trans_cost: float,
        max_weight: float,
        solver: str,
        inaccurate: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Method to run the MVO model over given periods
    """
    p_points = len(mu_lst)     # number of periods 

    assets = test_ret.columns  # names of all assets

    # LIST TO STORE VOLATILITY TARGETS
    list_portfolio_vty = []
    # LIST TO STORE VALUE OF THE PORTFOLIO
    list_portfolio_value = []
    # LIST TO STORE PORTFOLIO ALLOCATION
    list_portfolio_allocation = []

    x_old = pd.Series(0, index=assets)
    cash = budget
    portfolio_value_w = budget

    logger.debug(f"Selected solver is {solver}")
    for p in range(p_points):
        logger.info(f"Optimizing period {p}.")

        # Get MVO parameters
        mu = mu_lst[p]
        sigma = sigma_lst[p]

        # run CVaR model
        p_alloc, vty_val, port_val, cash = rebalancing_model(
            mu=mu,
            covariance=sigma,
            vty_targets=targets.loc[p, "Vty_Target"] * portfolio_value_w,
            cash=cash,
            x_old=x_old,
            trans_cost=trans_cost,
            max_weight=max_weight,
            solver=solver,
            inaccurate=inaccurate
        )

        # save the result
        list_portfolio_vty.append(vty_val)
        # save allocation
        list_portfolio_allocation.append(p_alloc)

        portfolio_value_w = port_val
        # COMPUTE PORTFOLIO VALUE
        for w in test_ret.index[(p * 4): (4 + p * 4)]:
            portfolio_value_w = sum(p_alloc * portfolio_value_w * (1 + test_ret.loc[w, assets]))
            list_portfolio_value.append((w, portfolio_value_w))

        x_old = p_alloc * portfolio_value_w

    portfolio_vty = pd.DataFrame(columns=["Volatility"], data=list_portfolio_vty)
    portfolio_value = pd.DataFrame(columns=["Date", "Portfolio_Value"], data=list_portfolio_value).set_index("Date", drop=True)
    portfolio_allocation = pd.DataFrame(columns=assets, data=list_portfolio_allocation)

    return portfolio_allocation, portfolio_value, portfolio_vty
