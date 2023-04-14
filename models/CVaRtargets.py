import pulp
import numpy as np
import pandas as pd
from loguru import logger

# FUNCTION RUNNING THE OPTIMIZATION
# ----------------------------------------------------------------------
def portfolio_risk_target(scenarios, cvar_alpha):
    
    # Fixed x
    x = pd.DataFrame(columns=scenarios.columns, index=["1/N position"])
    x.loc["1/N position", :] = 1/len(scenarios.columns)   
    
    # define index
    i_idx = scenarios.columns
    j_idx = scenarios.index
    
    # number of scenarios
    scenario_n = scenarios.shape[0]    
    
    # loss deviation
    var_dev = pulp.LpVariable.dicts("VarDev", (t for t in j_idx), lowBound=0, cat='Continuous')
        
    # value at risk
    var = pulp.LpVariable("VaR", lowBound=0, cat='Continuous')
    cvar = pulp.LpVariable("CVaR", lowBound=0, cat='Continuous')
      
    # define model
    model = pulp.LpProblem("Targets Optimization", pulp.LpMinimize)
     
    # Objective Function
    model += cvar
                      
    # *** CONSTRAINS ***               
    # Var deviation constrain
    for t in j_idx:
        model += -pulp.lpSum([scenarios.loc[t, i] * x[i] for i in i_idx]) - var <= var_dev[t]
    
    # CVaR constrain
    model += var + 1/(scenario_n * cvar_alpha) * pulp.lpSum([var_dev[t] for t in j_idx]) == cvar

    # Budget constrain
    model += pulp.lpSum([x[i] for i in i_idx]) == 1

    # solve model
    model.solve()
        
    # Get positions    
    if pulp.LpStatus[model.status] == 'Optimal':
     
        # print variables
        var_model = dict()
        for variable in model.variables():
            var_model[variable.name] = variable.varValue
         
        # solution with variable names   
        var_model = pd.Series(var_model, index=var_model.keys())
        
        return var_model["CVaR"]
    else:
        logger.exception(f"LP does not find optimal solution for CVaR targets with: {pulp.LpStatus[model.status]}")


# ----------------------------------------------------------------------
# Mathematical Optimization: TARGETS GENERATION
# ---------------------------------------------------------------------- 
def get_cvar_targets(test_date, benchmark, budget, cvar_alpha, data, scgen):

    # Define Benchmark
    tickers = benchmark

    # *** For Morningstar data ***
    target_weekly_ret = data[tickers].copy()

    # Get weekly data for testing
    test_weekly_ret = target_weekly_ret[target_weekly_ret.index >= test_date]

    # Number of weeks for testing
    weeks_n = len(test_weekly_ret.index)

    # Get scenarios
    # The Monte Carlo Method
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

if __name__ == "__main__":
    from pathlib import Path
    import os 
    import numpy as np

    from ScenarioGeneration import ScenarioGenerator

    ROOT_DIR = Path(__file__).parent.parent

    test_rng = np.random.default_rng(0)
    sg = ScenarioGenerator(test_rng)

    # Load our data
    weeklyReturns = pd.read_parquet(os.path.join(ROOT_DIR, 'financial_data/all_etfs_rets.parquet.gzip'))
    tickers = weeklyReturns.columns.values
    names = pd.read_parquet(os.path.join(ROOT_DIR, 'financial_data/all_etfs_rets_name.parquet.gzip')).columns.values

    start_test_date = pd.to_datetime("2017-07-01")
    end_test_date = pd.to_datetime("2022-07-20")

    # Find Benchmarks' ISIN codes
    benchmarks = ['iShares MSCI All Country Asia ex Japan Index Fund ETF', 'iShares MSCI ACWI ETF']
    benchmark_isin = [tickers[list(names).index(name)] for name in benchmarks]

    test_dataset = weeklyReturns[(weeklyReturns.index > start_test_date) & (weeklyReturns.index <= end_test_date)].copy()

    start_of_test_dataset = str(test_dataset.index.date[0])
    targets, benchmark_port_val = get_cvar_targets(
        test_date=start_of_test_dataset,
        benchmark=benchmark_isin, 
        budget=100,
        cvar_alpha=0.05,
        data=weeklyReturns,
        scgen=sg
    )

    targets.to_csv("targets_BASE.csv")
    benchmark_port_val.to_csv("benchmark_port_val_BASE.csv")