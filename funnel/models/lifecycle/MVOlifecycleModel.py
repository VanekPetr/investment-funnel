import math

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy as sp
from loguru import logger

"""'
Hvad med at vi splitter dataet f.eks. 70/30 og så sampler på de 70% og estimerer parametrer på de 30%
"""


def calculate_risk_metrics(yearly_returns, risk_free_rate=0.02):
    annual_return = yearly_returns.mean()
    annual_std_dev = yearly_returns.std()

    sharpe_ratio = (
        (annual_return - risk_free_rate) / annual_std_dev
        if annual_std_dev != 0
        else None
    )

    # Calculate downside deviation
    downside_diffs = [min(0, return_ - risk_free_rate) for return_ in yearly_returns]
    downside_std_dev = np.std(downside_diffs)

    # Calculate Sortino ratio
    sortino_ratio = (
        (annual_return - risk_free_rate) / downside_std_dev
        if downside_std_dev != 0
        else None
    )

    return annual_return, annual_std_dev, sharpe_ratio, downside_std_dev, sortino_ratio


def calculate_analysis_metrics(terminal_values):
    mean_terminal_value = np.mean(terminal_values)
    stdev_terminal_value = np.std(terminal_values)
    max_terminal_value = np.max(terminal_values)
    min_terminal_value = np.min(terminal_values)

    # Sorting for decile and quartile calculations
    sorted_terminal_values = sorted(terminal_values)
    lower_decile_count = len(sorted_terminal_values) // 10
    upper_decile_count = len(sorted_terminal_values) - lower_decile_count
    lower_quartile_count = len(sorted_terminal_values) // 4
    upper_quartile_count = len(sorted_terminal_values) - lower_quartile_count

    lower_decile_avg = np.mean(sorted_terminal_values[:lower_decile_count])
    upper_decile_avg = np.mean(sorted_terminal_values[-upper_decile_count:])
    lower_quartile_avg = np.mean(sorted_terminal_values[:lower_quartile_count])
    upper_quartile_avg = np.mean(sorted_terminal_values[-upper_quartile_count:])

    # Creating a DataFrame to store the metrics
    metrics_df = pd.DataFrame(
        {
            "Mean Terminal Value": [mean_terminal_value],
            "Standard Deviation Terminal Value": [stdev_terminal_value],
            "Max Terminal Value": [max_terminal_value],
            "Min Terminal Value": [min_terminal_value],
            "Lower Decile Average": [lower_decile_avg],
            "Upper Decile Average": [upper_decile_avg],
            "Lower Quartile Average": [lower_quartile_avg],
            "Upper Quartile Average": [upper_quartile_avg],
        }
    )

    return metrics_df


def cholesky_psd(m):
    """
    Computes the Cholesky decomposition of the given matrix, that is not positive definite, only semidefinite.
    """
    lu, d, perm = sp.linalg.ldl(m)
    assert np.max(np.abs(d - np.diag(np.diag(d)))) < 1e-8, "Matrix 'd' is not diagonal!"

    # Do non-negativity fix
    min_eig = np.min(np.diag(d))
    if min_eig < 0:
        d -= 5 * min_eig * np.eye(*d.shape)

    sqrtd = sp.linalg.sqrtm(d)
    C = (lu @ sqrtd).T
    return C


def lifecycle_rebalance_model(
    mu, sigma, vol_target, max_weight, solver, inaccurate: bool = True, lower_bound=0
):
    N = len(mu)  # Number of assets
    cash_index = mu.index.get_loc("Cash")  # Identify the index of the 'Cash' asset
    non_cash_indices = np.array([i for i in range(len(mu)) if i != cash_index])

    x = cp.Variable(
        N, name="x", nonneg=True
    )  # The weights of the assets (to be optimized)

    G = cholesky_psd(sigma)  # Transform to standard deviation matrix

    objective = cp.Maximize(x @ mu)

    constraints = [
        cp.sum(x) == 1,
        cp.norm(G @ x, 2) <= vol_target,
        # x <= max_weight * cp.sum(x)
        x[non_cash_indices] <= max_weight * cp.sum(x[non_cash_indices]),
    ]

    if lower_bound != 0:
        z = cp.Variable(
            N, boolean=True
        )  # Binary variable indicates if asset is selected
        upper_bound = 100

        constraints.append(lower_bound * z <= x)
        constraints.append(x <= upper_bound * z)
        constraints.append(cp.sum(z) >= 1)

    model = cp.Problem(objective, constraints)
    model.solve(solver=solver, qcp=True)

    # Get positions
    accepted_statuses = ["optimal"]
    if inaccurate:
        accepted_statuses.append("optimal_inaccurate")
    if model.status in accepted_statuses:
        port_nom = pd.Series(
            x.value, index=mu.index
        )  # Nominal allocations in each asset
        port_val = np.sum(port_nom)  # Portfolio value
        port_nom[np.abs(port_nom) < 0.000001] = 0  # Remove excessive noise.
        return port_nom, port_val

    else:
        # Print an error if the model is not optimal
        logger.exception(f"It's not looking good lad. Status code is {model.status}")
        port_nom = pd.Series(
            x.value, index=mu.index
        )  # Nominal allocations in each asset
        port_val = np.sum(port_nom)
        return port_nom, port_val


def get_port_allocations(mu_lst, sigma_lst, targets, max_weight, solver):
    """
    Parameters:
    - scen: Market scenarios
    - mu_lst, sigma_lst: Expected returns and standard deviations for each year
    - targets: Target volatilities
    - budget: Initial budget
    - trans_cost: Transaction costs
    - max_weight: Maximum weight for any asset
    - withdrawal_lst: List of withdrawals for each year
    - solver: Optimization solver to use
    - interest_rate: Interest rate for borrowing

    Returns:
    - ptf_df: DataFrame with portfolio values and other metrics for each year
    - allocation_df: DataFrame with asset allocations for each year
    """

    # Initial setup
    num_years = len(targets)
    years = [str(i + 2023) for i in range(num_years)]
    assets = mu_lst.index

    # Initialize DataFrames
    allocation_df = pd.DataFrame(index=years, columns=assets)

    for year in range(num_years):
        # logger.info(f'Rebalancing portfolio in the beginning of year {year + 2023}')

        port_weights, _ = lifecycle_rebalance_model(
            mu=mu_lst,
            sigma=sigma_lst,
            vol_target=targets.loc[year],
            max_weight=max_weight,
            solver=solver,
        )

        allocation_df.loc[allocation_df.index[year], :] = port_weights

    logger.info(
        f"The optimal portfolio allocations has been obtained for the {num_years+1} years."
    )
    return allocation_df


def lifecycle_model(
    alloc_targets, trans_cost, x_old, cash, solver, inaccurate: bool = True
):
    assets = alloc_targets.index
    N = len(assets)

    c = trans_cost  # Variable transaction costs
    x = cp.Variable(
        N, name="x", nonneg=True
    )  # The weights of the assets (to be optimized)
    absdiff = cp.Variable(N, name="absdiff", nonneg=True)  # - |x - x_old|
    cost = cp.Variable(name="cost", nonneg=True)  # - cost

    objective = cp.Minimize(cp.sum(absdiff))

    constraints = [
        x - x_old <= absdiff,
        x - x_old >= -absdiff,
        c * cp.sum(absdiff) == cost,
        x_old.sum() + cash - cp.sum(x) - cost == 0,
    ]

    for i in range(N):
        constraints.append(x[i] <= alloc_targets.loc[assets[i]])

    model = cp.Problem(objective, constraints)
    model.solve(solver=solver, qcp=True)

    # Get positions
    accepted_statuses = ["optimal"]
    if inaccurate:
        accepted_statuses.append("optimal_inaccurate")
    if model.status in accepted_statuses:
        port_nom = pd.Series(
            x.value, index=alloc_targets.index
        )  # Nominal allocations in each asset
        port_val = np.sum(port_nom)  # Portfolio value
        opt_port = (
            port_nom / port_val
            if port_val > 0
            else pd.Series(0, index=alloc_targets.columns)
        )  # Percentage allocation in each asset
        cash = cash - (
            port_val + cost.value - x_old.sum()
        )  # Update cash accordingly in each period
        opt_port[np.abs(opt_port) < 0.00001] = 0  # Remove excessive noise.
        return (
            opt_port,
            port_val,
            cost.value,
            cash,
            absdiff.value,
            absdiff.value.sum(),
            "optimal",
        )

    else:
        # Print an error if the model is not optimal
        logger.exception(f"It's not looking good lad. Status code is {model.status}")

        return None, None, None, cash, None, None, "infeasible"


def rebalancing_model_risk(
    scen, targets, budget, trans_cost, withdrawal_lst, interest_rate, solver
):
    """
    Parameters:
    - scen: Market scenarios
    - mu_lst, sigma_lst: Expected returns and standard deviations for each year
    - targets: Target volatilities
    - budget: Initial budget
    - trans_cost: Transaction costs
    - max_weight: Maximum weight for any asset
    - withdrawal_lst: List of withdrawals for each year
    - solver: Optimization solver to use
    - interest_rate: Interest rate for borrowing

    Returns:
    - ptf_df: DataFrame with portfolio values and other metrics for each year
    - allocation_df: DataFrame with asset allocations for each year
    """

    # Initial setup
    num_years = scen.shape[0]
    years = [str(i + 2023) for i in range(num_years)]
    assets = targets.columns  # [0].index

    # Initialize DataFrames
    ptf_columns = [
        "Portfolio_Value",
        "Costs",
        "Withdrawal",
        "Absolute DKK rebalanced",
        "Return",
        "Borrowed_Amount",
        "Yearly Returns",
    ]
    ptf_df = pd.DataFrame(index=years, columns=ptf_columns)
    allocation_df = pd.DataFrame(index=years, columns=assets)
    yearly_returns = []

    infeasible_period, borrowed_amount, interest_for_the_year = 0, 0, 0
    x_old, portfolio_value, cash = pd.Series(0, index=assets), budget, budget

    for year in range(num_years):
        # logger.info(f'Rebalancing portfolio in the beginning of year {year + 2023}')

        if infeasible_period > 0:
            withdrawal_amount = withdrawal_lst[year]
            borrowed_amount += withdrawal_amount
            interest_for_the_year = borrowed_amount * interest_rate
            borrowed_amount += interest_for_the_year  # Accumulate interest
            # Update DataFrame for infeasible period
            ptf_df.loc[ptf_df.index[year]] = {
                "Portfolio_Value": -borrowed_amount,
                "Costs": 0,
                "Standard Deviation": 0,
                "Withdrawal": withdrawal_amount,
                "Absolute DKK rebalanced": 0,
                "Return": -interest_for_the_year,
                "Borrowed_Amount": borrowed_amount - interest_for_the_year,
            }
            ptf_df["infeasible_period"] = infeasible_period
            continue

        if withdrawal_lst[year] >= portfolio_value * (1 + trans_cost):
            withdrawal_amount = portfolio_value
        else:
            withdrawal_amount = withdrawal_lst[year]
        cash -= math.floor(withdrawal_amount * (1 - trans_cost))

        (
            port_weights,
            port_val,
            port_costs,
            cash,
            absdiff_alloc,
            absdiff_total,
            status,
        ) = lifecycle_model(
            alloc_targets=targets.loc[f"{2023 + year}"] * portfolio_value,
            trans_cost=trans_cost,
            x_old=x_old,
            cash=cash,
            solver=solver,
        )

        if withdrawal_amount == portfolio_value:
            infeasible_period = year + 2023
            borrowed_amount = withdrawal_lst[year] - portfolio_value
            interest_for_the_year = borrowed_amount * interest_rate
            borrowed_amount += interest_for_the_year  # Accumulate interest

        year_return = np.dot(scen.loc[year], port_weights)
        yearly_returns.append(year_return)

        portfolio_value = port_val * (1 + year_return)
        x_old = port_weights * portfolio_value

        allocation_df.loc[allocation_df.index[year], :] = port_weights

        ptf_df.loc[allocation_df.index[year]] = {
            "Portfolio_Value": portfolio_value - borrowed_amount,
            "Costs": port_costs,
            "Withdrawal": withdrawal_lst[year],
            "Absolute DKK rebalanced": absdiff_total,
            "Return": port_val * year_return - interest_for_the_year,
            "Borrowed_Amount": borrowed_amount - interest_for_the_year,
            "Yearly Returns": year_return,
        }

        ptf_df["infeasible_period"] = infeasible_period

    return ptf_df, allocation_df


def riskadjust_model_scen(
    scen, targets, budget, trans_cost, withdrawal_lst, interest_rate, solver
):
    s_points, p_points, a_points = scen.shape
    assets = targets.columns
    # Initialize DataFrame
    portfolio_df = pd.DataFrame(
        columns=[
            "Terminal Wealth",
            "Total Returns",
            "Total Costs",
            "Total Withdrawals",
            "Infeasible Period",
            "Average Cash Hold",
            "Annual StDev",
            "Annual StDev_dd",
            "Average Annual Return",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Total borrowed",
        ],
        index=range(s_points),
    )
    all_allocations = np.zeros((s_points, p_points, a_points))

    for scenario in range(s_points):
        # logger.info(f"Optimizing for scenario {scenario}")
        scenarios_df = pd.DataFrame(scen[scenario, :, :], columns=assets)

        res, res_alloc = rebalancing_model_risk(
            scen=scenarios_df,
            targets=targets,
            budget=budget,
            trans_cost=trans_cost,
            withdrawal_lst=withdrawal_lst,
            interest_rate=interest_rate,
            solver=solver,
        )

        # Store the allocation results for this scenario
        all_allocations[scenario] = res_alloc.to_numpy()
        annual_return, annual_std_dev, sr, downside_std_dev, sortino_ratio = (
            calculate_risk_metrics(res["Yearly Returns"])
        )

        portfolio_df.loc[scenario] = {
            "Terminal Wealth": res["Portfolio_Value"].iloc[-1],
            "Total Returns": res["Return"].sum(),
            "Total Costs": res["Costs"].sum(),
            "Total Withdrawals": res["Withdrawal"].sum(),
            "Infeasible Period": res["infeasible_period"].max(),
            "Average Cash Hold": res_alloc["Cash"].mean(),
            "Annual StDev": annual_std_dev,
            "Annual StDev_dd": downside_std_dev,
            "Average Annual Return": annual_return,
            "Sharpe Ratio": sr,
            "Sortino Ratio": sortino_ratio,
            "Total borrowed": res["Borrowed_Amount"].sum(),
        }

        if scenario % (s_points // 2) == 0 and scenario != 0:
            logger.info(f"{scenario} out of {s_points} scenarios finished")
    non_zero_count = (portfolio_df["Infeasible Period"] != 0).sum()

    # Create a MultiIndex representing each scenario-period combination
    index = pd.MultiIndex.from_product(
        [range(s_points), range(p_points)], names=["scenario", "period"]
    )
    # Reshape the allocations data to a long format
    allocations_long = all_allocations.reshape(
        -1, a_points
    )  # a_points is the number of assets
    # Create the DataFrame with the MultiIndex
    allocations_df = pd.DataFrame(allocations_long, index=index, columns=assets)

    # Group by the 'period' level of the index and calculate mean
    mean_allocations_df = allocations_df.groupby("period").mean()
    analysis_metrics = calculate_analysis_metrics(portfolio_df["Terminal Wealth"])

    logger.info(
        f"{s_points} out of {s_points} scenarios has now been made. We saw a total of {non_zero_count} "
        f"incidents where the model is infeasible."
    )
    return portfolio_df, mean_allocations_df, analysis_metrics
