import pulp
import pandas as pd
from loguru import logger


# ----------------------------------------------------------------------
# MODEL FOR OPTIMIZING THE BACKTEST PERIODS 
# ----------------------------------------------------------------------
def rebalancing_model(mu, scenarios, cvar_targets, cvar_alpha, budget, x_old, trans_cost, max_weight, first=False):
    
    """ This function finds the optimal enhanced index portfolio according to some benchmark.
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
    budget:
        additional budget for our portfolio
    x_old:
        old portfolio allocation
    trans_cost:
        transaction costs
    max_weight : float
        Maximum allowed weight    
    cvar_alpha : float
        Alpha value used to evaluate Value-at-Risk one    
    
    Returns
    -------
    float
        Asset weights in an optimal portfolio 
    """ 
    # define index
    i_idx = scenarios.columns
    j_idx = scenarios.index
    
    # number of scenarios
    scenario_n = scenarios.shape[0]    
    # variable transaction costs
    c = trans_cost
    
    # define variables
    x = pulp.LpVariable.dicts("x", (i for i in i_idx), lowBound=0, cat='Continuous')
    
    # define variables for buying
    buy = pulp.LpVariable.dicts("buy", (i for i in i_idx), lowBound=0, cat='Continuous')
    # define variables for selling
    sell = pulp.LpVariable.dicts("sell", (i for i in i_idx), lowBound=0, cat='Continuous')
    
    # define cost variable
    cost = pulp.LpVariable("cost", lowBound=0, cat='Continuous') 

    # loss deviation
    var_dev = pulp.LpVariable.dicts("VarDev", (t for t in j_idx), lowBound=0, cat='Continuous')
        
    # value at risk
    var = pulp.LpVariable("VaR", lowBound=0, cat='Continuous')
    cvar = pulp.LpVariable("CVaR", lowBound=0, cat='Continuous')

    # *** define model ***
    model = pulp.LpProblem("Mean-CVaR Optimization", pulp.LpMaximize)
                      
    # *** Objective Function, maximize expected return of the portfolio ***         
    model += pulp.lpSum([mu[i] * x[i] for i in i_idx])

    # *** constraints ***
    # calculate VaR deviation
    for t in j_idx:
        model += -pulp.lpSum([scenarios.loc[t, i] * x[i] for i in i_idx]) - var <= var_dev[t]

    # calculate CVaR
    model += var + 1/(scenario_n * cvar_alpha) * pulp.lpSum([var_dev[t] for t in j_idx]) == cvar
    
    # CVaR target
    model += cvar <= cvar_targets     
    
    if first: 
        # budget constrain
        model += pulp.lpSum([x[i] for i in i_idx]) == (1-c) * budget
    else:
        # re-balancing
        for i in i_idx:
            model += x_old[i] - sell[i] + buy[i] == x[i]
        # cost of re-balancing
        model += c * (pulp.lpSum([buy[i] for i in i_idx]) + pulp.lpSum([sell[i] for i in i_idx])) == cost
        
        # new budget constrain
        model += pulp.lpSum([buy[i] for i in i_idx]) == pulp.lpSum([sell[i] for i in i_idx]) - cost

    # *** Concentration limits ***
    # set max limits, so it cannot not be larger than a fixed value
    if first:
        for i in i_idx:
            model += x[i] <= max_weight * (1-c)*budget
    else:
        for i in i_idx:
            model += x[i] <= max_weight * (x_old.sum() - cost)

    # *** solve model ***
    model.solve()
    
    # print an error if the model is not optimal
    if pulp.LpStatus[model.status] != 'Optimal':
        print("Whoops! There is an error! The model has error status:" + pulp.LpStatus[model.status])

    # *** Get positions ***
    if pulp.LpStatus[model.status] == 'Optimal':
     
        # print variables
        var_model = dict()
        for variable in model.variables():
            var_model[variable.name] = variable.varValue
         
        # solution with variable names   
        var_model = pd.Series(var_model, index=var_model.keys())

        long_pos = [i for i in var_model.keys() if i.startswith("x")]
             
        # total portfolio with negative values as short positions
        port_total = pd.Series(var_model[long_pos].values, index=[t[2:] for t in var_model[long_pos].index])
    
        opt_port = port_total
    
        # set floating data points to zero and normalize
        opt_port[opt_port < 0.000001] = 0
        cvar_result_p = var_model["CVaR"]/sum(opt_port)
        port_val = sum(opt_port)
        opt_port = opt_port/sum(opt_port)
        
        # return portfolio, CVaR, and alpha
        return opt_port, cvar_result_p, port_val
    
    else:
        logger.exception(f"Linear solver does not find optimal solution with status code {pulp.LpStatus[model.status]}")


# ----------------------------------------------------------------------
# Mathematical Optimization: RUN THE CVAR MODEL
# ----------------------------------------------------------------------
def cvar_model(test_ret, scenarios, targets, budget, cvar_alpha, trans_cost, max_weight):
    """
    Method to run the CVaR model over given periods
    """
    p_points = len(scenarios[:, 0, 0])           # number of periods
    s_points = len(scenarios[0, :, 0])           # number of scenarios
    prob = 1/s_points                       # probability of each scenario

    assets = test_ret.columns                # names of all assets

    # LIST TO STORE CVaR TARGETS
    list_portfolio_cvar = []
    # LIST TO STORE VALUE OF THE PORTFOLIO
    list_portfolio_value = []
    # LIST TO STORE PORTFOLIO ALLOCATION
    list_portfolio_allocation = []

    for p in range(p_points):
        # Create dataframe with scenarios for a period p
        scenarios_df = pd.DataFrame(scenarios[p, :, :], columns=test_ret.columns)

        # compute expected returns of all assets (EP)
        expected_returns = sum(prob * scenarios_df.loc[i, :] for i in scenarios_df.index)

        if p == 0:
            # run CVaR model
            p_alloc, cvar_val, port_val = rebalancing_model(mu=expected_returns,
                                                            scenarios=scenarios_df,
                                                            cvar_targets=targets.loc[0, "CVaR_Target"] * budget,
                                                            cvar_alpha=cvar_alpha,
                                                            budget=budget,
                                                            x_old=None,
                                                            trans_cost=trans_cost,
                                                            max_weight=max_weight, 
                                                            first=True)

        if p > 0:
            # run CVaR model
            p_alloc, cvar_val, port_val = rebalancing_model(mu=expected_returns,
                                                            scenarios=scenarios_df,
                                                            cvar_targets=targets.loc[p, "CVaR_Target"] * portfolio_value_w,
                                                            cvar_alpha=cvar_alpha,
                                                            budget=None,
                                                            x_old=p_alloc * portfolio_value_w,
                                                            trans_cost=trans_cost,
                                                            max_weight=max_weight)
            
        # save the result
        list_portfolio_cvar.append(cvar_val)
        # save allocation
        list_portfolio_allocation.append(p_alloc)

        portfolio_value_w = port_val
        # COMPUTE PORTFOLIO VALUE
        for w in test_ret.index[(p * 4): (4 + p * 4)]:
            portfolio_value_w = sum(p_alloc * portfolio_value_w * (1 + test_ret.loc[w, assets]))
            list_portfolio_value.append((w, portfolio_value_w))

    portfolio_cvar = pd.DataFrame(columns=["CVaR"], data=list_portfolio_cvar)
    portfolio_value = pd.DataFrame(columns=["Date", "Portfolio_Value"], data=list_portfolio_value).set_index("Date", drop=True)
    portfolio_allocation = pd.DataFrame(columns=assets, data=list_portfolio_allocation)

    return portfolio_allocation, portfolio_value, portfolio_cvar
