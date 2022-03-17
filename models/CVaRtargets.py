
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 08:00:08 2020

@author: Petr Vanek
"""

import pulp
import pandas as pd
from models.ScenarioGeneration import BOOT

# FUNCTION RUNNING THE OPTIMIZATION
#---------------------------------------------------------------------- 
def PortfolioRiskTarget(scen, cvar_alpha):
    
    # Fixed x
    x = pd.DataFrame(columns=scen.columns,index=["1/N position"])
    x.loc["1/N position",:] = 1/len(scen.columns)   
    
    
    # define index
    i_idx = scen.columns
    j_idx = scen.index
    
    # number of scenarios
    N = scen.shape[0]    
    
    
    # loss deviation
    VarDev = pulp.LpVariable.dicts("VarDev", ( (t) for t in j_idx ),
                                   lowBound=0, cat='Continuous')
        
    # value at risk
    VaR = pulp.LpVariable("VaR", lowBound=0, cat='Continuous')
    CVaR = pulp.LpVariable("CVaR", lowBound=0, cat='Continuous')
    
        
    #####################################
    # define model
    model = pulp.LpProblem("Targets Optimization", pulp.LpMinimize)
     
    #####################################
    # Objective Function
    model += CVaR
                      
    #####################################
    # constraints
                      
    # Var deviation constrain
    for t in j_idx:
        model += -pulp.lpSum([scen.loc[t, i] * x[i] for i in i_idx]) - VaR <= VarDev[t]
    
    # CVaR constrain
    model += VaR + 1/(N*cvar_alpha)*pulp.lpSum([VarDev[t] for t in j_idx]) == CVaR

    # Budget constrain
    model += pulp.lpSum([x[i] for i in i_idx]) == 1

    # solve model
    model.solve()
    
    # print an error if the model is not optimal
    if pulp.LpStatus[model.status] != 'Optimal':
        print("Whoops! There is an error! The model has error status:" + pulp.LpStatus[model.status] )
        
    
    #Get positions    
    if pulp.LpStatus[model.status] == 'Optimal':
     
        # print variables
        var_model = dict()
        for variable in model.variables():
            var_model[variable.name] = variable.varValue
         
        # solution with variable names   
        var_model = pd.Series(var_model,index=var_model.keys())
         
    
    
    # return portfolio, CVaR, and alpha
    return var_model["CVaR"]


"""
    ----------------------------------------------------------------------
    Mathematical Optimization: TARGETS GENERATION
    ---------------------------------------------------------------------- 
"""

def targetsCVaR(start_date, end_date, test_date, benchmark, test_index, budget, cvar_alpha, data=[]):

    # Define Benchmark
    tickers = benchmark

    ## For YAHOO version
    # # User pandas_reader.data.DataReader to load the desired data.
    # panel_data = data.DataReader(tickers, 'yahoo', start_date, end_date)
    # df_close = panel_data["Adj Close"]
    # # Get weekly data
    # targetWeeklyRet = getWeeklyRet(data=df_close)

    # df_test_index = pd.DataFrame(index=test_index)
    # testWeeklyRet = pd.concat([df_test_index, testWeeklyRet], axis=1)
    # testWeeklyRet = testWeeklyRet.fillna(0)

    ## For Morningstar data
    targetWeeklyRet = data[tickers].copy()

    # Get weekly data for testing
    testWeeklyRet = targetWeeklyRet[targetWeeklyRet.index >= test_date]

    # Number of weeks for testing
    N_test = len(testWeeklyRet.index)

    # Get scenarios
    # The Monte Carlo Method
    targetScen = BOOT(data=targetWeeklyRet,       # subsetMST or subsetCLUST
                      n_simulations=250,
                      n_test=N_test)

    # Compute the optimal portfolio outperforming zero percentage return
    # ----------------------------------------------------------------------
    p_points = len(targetScen[:,0,0])       # number of periods
    s_points = len(targetScen[0,:,0])       # number of scenarios

    # DATA FRAME TO STORE CVaR TARGETS
    targets = pd.DataFrame(columns=["CVaR_Target"], index=list(range(p_points)))
    # DATA FRAME TO STORE VALUE OF THE PORTFOLIO
    portValue = pd.DataFrame(columns=["Benchmark_Value"], index=testWeeklyRet.index)

    # COMPUTE CVaR TARGETS
    for p in range(p_points):
        # create data frame with scenarios for a given period p
        scenDf = pd.DataFrame(targetScen[p,:,:],
                              columns=tickers,
                              index=list(range(s_points)))

        # run CVaR model to compute CVaR targets
        target_CVaR= PortfolioRiskTarget(scen=scenDf,
                                         cvar_alpha=cvar_alpha)
        # save the result
        targets.loc[p, "CVaR_Target"] = target_CVaR

    # COMPUTE PORTFOLIO VALUE
    for w in testWeeklyRet.index:
        portValue.loc[w, "Benchmark_Value"] = sum((budget/len(tickers))*(1+testWeeklyRet.loc[w, :]))
        budget = sum(budget/len(tickers)*(1+testWeeklyRet.loc[w, :]))

    
    return targets, portValue

