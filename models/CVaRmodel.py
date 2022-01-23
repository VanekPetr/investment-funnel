import pulp
import pandas as pd


# ----------------------------------------------------------------------
# MODEL FOR THE SECOND AND ONGOING PERIODS 
# ----------------------------------------------------------------------
def rebalancingModel(mu,scen,CVaR_target,cvar_alpha,x_old,trans_cost, max_weight):
    
    """ This function finds the optimal enhanced index portfolio according to some benchmark.
    The portfolio corresponds to the tangency portfolio where risk is evaluated according to 
    the CVaR of the tracking error. The model is formulated using fractional programming.
    
    Parameters
    ----------
    mu : pandas.Series with float values
        asset point forecast
    mu_b : pandas.Series with float values
        Benchmark point forecast
    scen : pandas.DataFrame with float values
        Asset scenarios
    scen_b : pandas.Series with float values
        Benchmark scenarios
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
    i_idx = scen.columns
    j_idx = scen.index
    
    # number of scenarios
    N = scen.shape[0]    
    # variable costs
    c = trans_cost
    
    # define variables
    x = pulp.LpVariable.dicts("x", ((i) for i in i_idx), lowBound=0, cat='Continuous')
    
    # define variables for buying
    buy = pulp.LpVariable.dicts("buy", ((i) for i in i_idx), lowBound=0, cat='Continuous')
    # define variables for selling
    sell = pulp.LpVariable.dicts("sell", ((i) for i in i_idx), lowBound=0, cat='Continuous')
    
    # define cost variable
    cost = pulp.LpVariable("cost", lowBound=0, cat='Continuous') 
    
    # loss deviation
    VarDev = pulp.LpVariable.dicts("VarDev", ((t) for t in j_idx), lowBound=0, cat='Continuous')
        
    # value at risk
    VaR = pulp.LpVariable("VaR", lowBound=0, cat='Continuous')
    CVaR = pulp.LpVariable("CVaR", lowBound=0, cat='Continuous')

    # *** define model ***
    model = pulp.LpProblem("Mean-CVaR Optimization", pulp.LpMaximize)

    # *** Objective Function, maximize expected return of the portfolio ***
             
    model += pulp.lpSum([mu[i] * x[i] for i in i_idx] )

    # *** constraints ***
                      
    # calculate VaR deviation
    for t in j_idx:
        model += -pulp.lpSum([scen.loc[t, i] * x[i] for i in i_idx]) - VaR <= VarDev[t]
    
    # calculate CVaR
    model += VaR + 1/(N*cvar_alpha)*pulp.lpSum([VarDev[t] for t in j_idx]) == CVaR
    
    # CVaR target
    model += CVaR <= CVaR_target     
    
    # re-balancing
    for t in i_idx:
        model += x_old[t] - sell[t] + buy[t] == x[t]
        
    # cost of re-balancing
    model += c * (pulp.lpSum([ buy[i] for i in i_idx]) + pulp.lpSum([sell[i] for i in i_idx])) == cost
    
    # new budget constrain
    model += pulp.lpSum([buy[i] for i in i_idx]) == pulp.lpSum([sell[i] for i in i_idx]) - cost
    
    # *** Concentration limits ***
    # set max limits, so it cannot not be larger than a fixed value
    ###
    for i in i_idx:
        model += x[i] <= max_weight*(x_old.sum()-cost)

    # *** solve model ***
    model.solve()
    
    # print an error if the model is not optimal
    if pulp.LpStatus[model.status] != 'Optimal':
        print("Whoops! There is an error! The model has error status:" + pulp.LpStatus[model.status] )

    # *** Get positions ***
    if pulp.LpStatus[model.status] == 'Optimal':
     
        # print variables
        var_model = dict()
        for variable in model.variables():
            var_model[variable.name] = variable.varValue
         
        # solution with variable names   
        var_model = pd.Series(var_model,index=var_model.keys())

        long_pos = [i for i in var_model.keys() if i.startswith("x")]
             
        # total portfolio with negative values as short positions
        port_total = pd.Series(var_model[long_pos].values, index=[t[2:] for t in var_model[long_pos].index])
    
        opt_port = port_total
    
    # set floating data points to zero and normalize
    opt_port[opt_port < 0.000001] = 0
    CVaR_result_p = var_model["CVaR"]/sum(opt_port)
    port_val = sum(opt_port)
    opt_port = opt_port/sum(opt_port)
    
    # return portfolio, CVaR, and alpha
    return opt_port, CVaR_result_p, port_val


# ----------------------------------------------------------------------
# MODEL FOR THE FIRST PERIOD 
# ----------------------------------------------------------------------
def firstPeriodModel(mu, scen, CVaR_target, cvar_alpha, budget, trans_cost, max_weight):
    
    """ This function finds the optimal enhanced index portfolio according to some benchmark.
    The portfolio corresponds to the tangency portfolio where risk is evaluated according to 
    the CVaR of the tracking error. The model is formulated using fractional programming.
    
    Parameters
    ----------
    mu : pandas.Series with float values
        asset point forecast
    mu_b : pandas.Series with float values
        Benchmark point forecast
    scen : pandas.DataFrame with float values
        Asset scenarios
    scen_b : pandas.Series with float values
        Benchmark scenarios
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
    i_idx = scen.columns
    j_idx = scen.index
    
    # number of scenarios
    N = scen.shape[0] 
    # variable transaction costs
    c = trans_cost
    
    # define variables
    x = pulp.LpVariable.dicts("x", ((i) for i in i_idx), lowBound=0, cat='Continuous')
    
    # loss deviation
    VarDev = pulp.LpVariable.dicts("VarDev", ((t) for t in j_idx), lowBound=0, cat='Continuous')
        
    # value at risk
    VaR = pulp.LpVariable("VaR", lowBound=0, cat='Continuous')
    CVaR = pulp.LpVariable("CVaR", lowBound=0, cat='Continuous')

    # *** define model ***
    model = pulp.LpProblem("Mean-CVaR Optimization", pulp.LpMaximize)

    # *** Objective Function, maximize expected return of the portfolio ***
             
    model += pulp.lpSum([mu[i] * x[i] for i in i_idx])

    #  *** constraints ***
    # budget constrain
    model += pulp.lpSum([x[i] for i in i_idx]) == (1-c) * budget
                      
    # calculate VaR deviation
    for t in j_idx:
        model += -pulp.lpSum([scen.loc[t, i] * x[i] for i in i_idx]) - VaR <= VarDev[t]
    
    # calculate CVaR
    model += VaR + 1/(N * cvar_alpha) * pulp.lpSum([VarDev[t] for t in j_idx]) == CVaR
    
    # CVaR target
    model += CVaR <= CVaR_target     
    
    # *** Concentration limits ***
    # set max limits, so it cannot not be larger than a fixed value
    for i in i_idx:
        model += x[i] <= max_weight*(1-c)*budget

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
        var_model = pd.Series(var_model,index=var_model.keys())

        long_pos = [i for i in var_model.keys() if i.startswith("x")]
             
        # total portfolio with negative values as short positions
        port_total = pd.Series(var_model[long_pos].values, index=[t[2:] for t in var_model[long_pos].index])
    
        opt_port = port_total
    
    # *** set floating data points to zero and normalize ***
    opt_port[opt_port < 0.000001] = 0
    CVaR_result_p = var_model["CVaR"]/sum(opt_port)
    port_val = sum(opt_port)
    opt_port = opt_port/sum(opt_port)
    
    # return portfolio, CVaR, and alpha
    return opt_port, CVaR_result_p, port_val


"""
    ----------------------------------------------------------------------
    Mathematical Optimization: RUN THE CVAR MODEL
    ----------------------------------------------------------------------
"""
def modelCVaR(testRet, scen, targets, budget, cvar_alpha, trans_cost, max_weight):
    """
    Method to run the CVaR model over given periods
    """
    p_points = len(scen[:, 0, 0])           # number of periods
    s_points = len(scen[0, :, 0])           # number of scenarios
    prob = 1/s_points                       # probability of each scenario

    assets = testRet.columns                # names of all assets

    # DATA FRAME TO STORE CVaR TARGETS
    portCVaR = pd.DataFrame(columns=["CVaR"], index=list(range(p_points)))
    # DATA FRAME TO STORE VALUE OF THE PORTFOLIO
    portValue = pd.DataFrame(columns=["Portfolio_Value"], index=testRet.index)
    # DATA FRAME TO STORE PORTFOLIO ALLOCATION
    portAllocation = pd.DataFrame(columns=assets, index=list(range(p_points)))

    # *** THE FIRST INVESTMENT PERIOD ***
    # ----------------------------------------------------------------------
    # create data frame with scenarios for a period p=0
    scenDf = pd.DataFrame(scen[0, :, :],
                          columns=testRet.columns,
                          index=list(range(s_points)))

    # compute expected returns of all assets
    EP = sum(prob*scenDf.loc[i, :] for i in scenDf.index)

    # run CVaR model
    p_alloc, CVaR_val, port_val = firstPeriodModel(mu=EP,
                                                   scen=scenDf,
                                                   CVaR_target=targets.loc[0, "CVaR_Target"] * budget,
                                                   cvar_alpha=cvar_alpha,
                                                   budget=budget,
                                                   trans_cost=trans_cost,
                                                   max_weight=max_weight)

    # save the result
    portCVaR.loc[0, "CVaR"] = CVaR_val
    # save allocation
    portAllocation.loc[0, assets] = p_alloc
    portValueW = port_val

    # COMPUTE PORTFOLIO VALUE
    for w in testRet.index[0:4]:
        portValue.loc[w,"Portfolio_Value"] = sum(portAllocation.loc[0, assets] * portValueW
                                                 * (1+testRet.loc[w, assets]))
        portValueW = portValue.loc[w, "Portfolio_Value"]

    # *** THE SECOND AND ONGOING INVESTMENT PERIODS ***
    # ----------------------------------------------------------------------
    for p in range(1, p_points):
        # create data frame with scenarios for a given period p
        scenDf = pd.DataFrame(scen[p, :, :],
                              columns=testRet.columns,
                              index=list(range(s_points)))  
    
        # compute expected returns of all assets
        EP = sum(prob*scenDf.loc[i, :] for i in scenDf.index)

        # run CVaR model
        p_alloc, CVaR_val, port_val = rebalancingModel(mu=EP,
                                                       scen=scenDf,
                                                       CVaR_target=targets.loc[p, "CVaR_Target"] * portValueW,
                                                       cvar_alpha=cvar_alpha,
                                                       x_old=portAllocation.loc[p-1, assets] * portValueW,
                                                       trans_cost=trans_cost,
                                                       max_weight=max_weight)
        # save the result
        portCVaR.loc[p, "CVaR"] = CVaR_val
        # save allocation
        portAllocation.loc[p, assets] = p_alloc

        portValueW = port_val
        # COMPUTE PORTFOLIO VALUE
        for w in testRet.index[(p * 4): (4 + p * 4)]:
            portValue.loc[w, "Portfolio_Value"] = sum(portAllocation.loc[p, assets]
                                                      * portValueW*(1+testRet.loc[w, assets]))
            portValueW = portValue.loc[w, "Portfolio_Value"]
    
    return portAllocation, portValue, portCVaR
