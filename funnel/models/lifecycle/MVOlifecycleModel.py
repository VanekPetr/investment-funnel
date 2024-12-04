from typing import List

import cvxpy as cp
import numpy as np
import pandas as pd
from loguru import logger

from ..MVOmodel import cholesky_psd


def calculate_risk_metrics(
    yearly_returns: pd.Series, risk_free_rate: float = 0.02
) -> (float, float, float, float, float):
    """Calculate risk metrics for a given set of yearly returns."""
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


def calculate_analysis_metrics(terminal_values: pd.Series) -> pd.DataFrame:
    """Calculate various metrics from the terminal values of the portfolio."""
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


def lifecycle_rebalance_model(
    mu: pd.DataFrame,
    sigma: pd.DataFrame,
    vol_target: float,
    max_weight: float,
    solver: str,
    inaccurate: bool = True,
    lower_bound: float = 0,
) -> (pd.Series, float):
    """
    Optimizes asset allocations within a portfolio to maximize expected returns
    while adhering to a risk budget glide path.

    Parameters:
    - mu: Expected returns for each asset.
    - sigma: Covariance matrix of asset returns.
    - vol_target: Target portfolio volatility for period.
    - max_weight: Maximum weight allowed for any single asset (excluding cash).
    - solver: Solver to be used by CVXPY for optimization.
    - inaccurate: If True, accepts 'optimal_inaccurate' as a successful solve status.

    Returns:
    - port_nom: Nominal allocations for each asset in the optimized portfolio.
    - port_val: Total value of the portfolio based on the allocations.

    This function uses convex optimization to find the asset weights that maximize
    expected returns subject to constraints on total weight, individual asset weights,
    and portfolio volatility. It optionally includes binary selection variables to enforce
    a minimum allocation to any selected asset.
    """

    # Prepare basic variables and indices
    N = len(mu)  # Number of assets
    cash_index = mu.index.get_loc("Cash")  # Identify the index of the 'Cash' asset
    non_cash_indices = np.array([i for i in range(len(mu)) if i != cash_index])

    # Optimization variables
    x = cp.Variable(N, name="x", nonneg=True)  # Asset weights

    # Prepare matrix for volatility constraint
    G = cholesky_psd(sigma)  # Transform to standard deviation matrix

    # Define the optimization problem
    objective = cp.Maximize(x.T @ mu)

    constraints = [
        cp.sum(x) == 1,  # Weights sum to 1
        cp.norm(G @ x, 2) <= vol_target,  # Portfolio volatility constraint
        # cp.quad_form(x, sigma) <= vol_target ** 2,
        x[non_cash_indices]
        <= max_weight
        * cp.sum(x[non_cash_indices]),  # Max weight constraint for non-cash assets
    ]

    # Optional lower bound constraint
    if lower_bound != 0:
        z = cp.Variable(
            N, boolean=True
        )  # Binary variable indicates if asset is selected
        upper_bound = 100  # Arbitrary upper bound for asset weights

        constraints += [
            lower_bound * z <= x,  # Lower bound constraint
            x <= upper_bound * z,  # Upper bound enables asset deselection
            cp.sum(z) >= 1,  # At least one asset must be selected
        ]

    # Solve the optimization problem
    model = cp.Problem(objective, constraints)
    model.solve(solver=solver, qcp=True)

    # Process optimization results
    accepted_statuses = ["optimal"]
    if inaccurate:
        accepted_statuses.append("optimal_inaccurate")
    if model.status in accepted_statuses:
        port_nom = pd.Series(
            x.value, index=mu.index
        )  # Nominal allocations in each asset
        port_val = np.sum(port_nom)  # Total portfolio value
        port_nom[np.abs(port_nom) < 0.00001] = (
            0  # Eliminate numerical noise by zeroing very small weights
        )
        return port_nom, port_val

    else:
        # Handle non-optimal solve status
        logger.exception(
            f"The model is {model.status}. Look into the constraints. It might be an issue of too low risk targets."
        )
        port_nom = pd.Series(np.nan, index=mu.index)  # Use NaNs to indicate failure
        port_val = np.sum(port_nom)
        return port_nom, port_val


def get_port_allocations(
    mu_lst: pd.DataFrame,
    sigma_lst: pd.DataFrame,
    targets: pd.DataFrame,
    max_weight: float,
    solver: str,
) -> pd.DataFrame:
    """
    Calculates optimal portfolio allocations for the glide paths.

    Parameters:
    - mu_lst, sigma_lst: Expected returns and standard deviations for each year
    - targets: Target volatilities from the risk budget glide paths
    - budget: Initial budget
    - trans_cost: Transaction costs
    - max_weight: Maximum weight for any asset
    - withdrawal_lst: List of withdrawals for each year
    - solver: Optimization solver to use
    - interest_rate: Interest rate for borrowing

    Returns:
    - allocation_df: DataFrame showing the optimal asset allocations for each year.

    This function iterates through each year, using the provided expected returns, covariance matrices,
    and risk budget glide path to determine the optimal asset allocations.
    """
    # Initial setup
    num_years = len(targets)
    years = [str(i + 2023) for i in range(num_years)]
    assets = mu_lst.index

    # Initialize DataFrames
    allocation_df = pd.DataFrame(index=years, columns=assets)

    for year in range(num_years):
        port_weights, _ = lifecycle_rebalance_model(
            mu=mu_lst,
            sigma=sigma_lst,
            vol_target=targets.loc[year],
            max_weight=max_weight,
            solver=solver,
        )

        allocation_df.loc[allocation_df.index[year], :] = port_weights

    logger.info(
        f"The optimal portfolio allocations has been obtained for the {num_years + 1} years."
    )
    return allocation_df


def portfolio_rebalancing(
    budget: float,
    targets: pd.DataFrame,
    withdrawal_lst: List[float],
    transaction_cost: float,
    scenarios: pd.DataFrame,
    interest_rate: float,
) -> (pd.DataFrame, pd.DataFrame):
    """
    Simulates portfolio rebalancing over multiple years, accounting for withdrawals,
    transaction costs, returns based on scenarios, and handling defaults when withdrawals exceed portfolio value.

    Parameters:
    - budget: Initial budget available for investment.
    - targets: DataFrame containing target asset allocations for each year.
    - withdrawal_lst: List of annual withdrawal amounts.
    - transaction_cost: Transaction cost rate applied to rebalancing and withdrawals.
    - scenarios: DataFrame containing asset return scenarios for each year.
    - interest_rate: Interest rate applied to borrowed amounts in case of default.

    Returns:
    - ptf_performance: DataFrame for portfolio performance for each year.
    - allocation_df: DataFrame showing asset allocations for each year.
    """

    # Initialize the number of years and assets from the targets DataFrame
    n_years, n_assets = targets.shape
    years = [i + 2023 for i in range(n_years)]
    assets = targets.columns

    # Prepare DataFrame to store portfolio performance
    ptf_performance_columns = [
        "Portfolio Value Primo",
        "Portfolio Value Ultimo",
        "Withdrawal",
        "Returns in DKK",
        "Yearly Returns",
        "Transaction Costs",
        "Absolute DKK Rebalanced",
        "Borrowed Amount",
    ]
    ptf_performance = pd.DataFrame(index=years, columns=ptf_performance_columns)

    # DataFrame for tracking asset allocations each year
    allocation_df = pd.DataFrame(0, index=years, columns=assets, dtype=float)

    # Initialize portfolio values
    default_year, borrowed_amount, interest_for_the_year = 0, 0, 0
    x_old, portfolio_value_ultimo_aw = pd.Series(0, index=targets.columns), budget

    for year in range(n_years):
        # Handle scenario where the portfolio is in default
        if default_year > 0:
            withdrawal_amount = withdrawal_lst[year]
            borrowed_amount += withdrawal_amount
            interest_for_the_year = borrowed_amount * interest_rate
            borrowed_amount += interest_for_the_year  # Accumulate interest

            # Update performance DataFrame for periods in default
            ptf_performance.loc[ptf_performance.index[year]] = {
                "Portfolio Value Primo": ptf_performance["Portfolio Value Ultimo"][
                    ptf_performance.index[year - 1]
                ],
                "Portfolio Value Ultimo": -borrowed_amount,
                "Withdrawal": withdrawal_lst[year],
                "Returns in DKK": -interest_for_the_year,
                "Yearly Returns": -interest_rate,
                "Transaction Costs": 0,
                "Absolute DKK Rebalanced": 0,
                "Borrowed Amount": borrowed_amount - interest_for_the_year,
            }
            ptf_performance["Default Year"] = default_year
            continue

        # Normal operation: calculate returns, rebalancing, and manage withdrawals
        port_weights = targets.iloc[year]
        absolute_rebalance = (
            port_weights - x_old
        ).abs().sum() * portfolio_value_ultimo_aw
        costs_rebalance = absolute_rebalance * transaction_cost

        portfolio_value_primo = portfolio_value_ultimo_aw - costs_rebalance

        year_return = np.dot(scenarios.loc[year], port_weights)
        portfolio_value_ultimo_bw = portfolio_value_primo * (1 + year_return)

        # Check if withdrawals exceed the portfolio value, leading to default
        if withdrawal_lst[year] >= portfolio_value_ultimo_bw * (1 + transaction_cost):
            withdrawal_amount = portfolio_value_ultimo_bw * (1 - transaction_cost)
            default_year = year + 2023
            borrowed_amount = withdrawal_lst[year] - withdrawal_amount
            interest_for_the_year = borrowed_amount * interest_rate
            borrowed_amount += interest_for_the_year

        else:
            withdrawal_amount = withdrawal_lst[year]

        costs_withdraw = withdrawal_amount * transaction_cost
        costs_total = costs_rebalance + costs_withdraw
        portfolio_value_ultimo_aw = portfolio_value_ultimo_bw - (
            withdrawal_amount + costs_withdraw
        )

        x_old = port_weights

        # Update allocation DataFrame for the year
        allocation_df.loc[allocation_df.index[year], :] = port_weights

        # Update summary DataFrame for normal operation
        ptf_performance.loc[ptf_performance.index[year]] = {
            "Portfolio Value Primo": portfolio_value_primo,
            "Portfolio Value Ultimo": portfolio_value_ultimo_aw - borrowed_amount,
            "Withdrawal": withdrawal_lst[year],
            "Returns in DKK": portfolio_value_primo * year_return,
            "Yearly Returns": year_return,
            "Transaction Costs": costs_total,
            "Absolute DKK Rebalanced": absolute_rebalance,
            "Borrowed Amount": borrowed_amount,
        }
        ptf_performance["Default Year"] = default_year

    return ptf_performance, allocation_df


def riskadjust_model_scen(
    scen: np.ndarray,
    targets: pd.DataFrame,
    budget: float,
    trans_cost: float,
    withdrawal_lst: List[float],
    interest_rate: float,
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Simulates portfolio performance across different scenarios, adjusting for risk and calculating
    various financial metrics based on the portfolio's rebalancing strategy.

    Parameters:
    - scen: 3D numpy array containing return scenarios (scenarios, periods, assets).
    - targets: DataFrame specifying target asset allocations for each period.
    - budget: Initial budget for investment.
    - trans_cost: Transaction costs as a fraction of the trade amount.
    - withdrawal_lst: List of annual withdrawal amounts.
    - interest_rate: Interest rate applied to borrowed amounts in case of default.

    Returns:
    - portfolio_df: DataFrame summarizing the performance metrics for each scenario.
    - mean_allocations_df: DataFrame showing the average asset allocations across all scenarios for each period.
    - analysis_metrics: Dictionary containing overall analysis metrics calculated from portfolio_df.

    The function iterates through each scenario, rebalances the portfolio according to the targets,
    and calculates performance metrics such as total returns, costs, withdrawals, and risk-adjusted measures.
    It also tracks the occurrence of default events and calculates average allocations and other analysis metrics.
    """

    s_points, p_points, a_points = (
        scen.shape
    )  # Scenario, periods, and assets dimensions
    assets = targets.columns  # Asset names from the targets DataFrame

    # Initialize DataFrame to hold portfolio performance metrics for each scenario
    portfolio_df = pd.DataFrame(
        columns=[
            "Terminal Wealth",
            "Total Returns",
            "Total Costs",
            "Total Withdrawals",
            "Default Year",
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
    # Initialize array to hold asset allocations for all scenarios
    all_allocations = np.zeros((s_points, p_points, a_points))

    for scenario in range(s_points):
        # Convert scenario data to DataFrame for processing
        scenarios_df = pd.DataFrame(scen[scenario, :, :], columns=assets)

        # Perform portfolio rebalancing for the scenario
        res, res_alloc = portfolio_rebalancing(
            budget=budget,
            targets=targets,
            withdrawal_lst=withdrawal_lst,
            transaction_cost=trans_cost,
            scenarios=scenarios_df,
            interest_rate=interest_rate,
        )

        # Store the allocation results for this scenario
        all_allocations[scenario] = res_alloc.to_numpy()

        # Calculate risk metrics for the scenario
        annual_return, annual_std_dev, sr, downside_std_dev, sortino_ratio = (
            calculate_risk_metrics(res["Yearly Returns"])
        )

        # Update the portfolio DataFrame with calculated metrics for the scenario
        portfolio_df.loc[scenario] = {
            "Terminal Wealth": res["Portfolio Value Ultimo"].iloc[-1],
            "Total Returns": res["Returns in DKK"].sum(),
            "Total Costs": res["Transaction Costs"].sum(),
            "Total Withdrawals": res["Withdrawal"].sum(),
            "Default Year": res["Default Year"].max(),
            "Average Cash Hold": res_alloc["Cash"].mean(),
            "Annual StDev": annual_std_dev,
            "Annual StDev_dd": downside_std_dev,
            "Average Annual Return": annual_return,
            "Sharpe Ratio": sr,
            "Sortino Ratio": sortino_ratio,
            "Total borrowed": res["Borrowed Amount"].sum(),
        }

        if scenario % (s_points // 2) == 0 and scenario != 0:
            logger.info(f"{scenario} out of {s_points} scenarios finished")

    # Log progress and calculate default count
    default_count = (portfolio_df["Default Year"] != 0).sum()

    # Reshape and aggregate allocation data for analysis
    index = pd.MultiIndex.from_product(
        [range(s_points), range(p_points)], names=["scenario", "period"]
    )
    allocations_long = all_allocations.reshape(-1, a_points)  # Reshape to long format
    allocations_df = pd.DataFrame(allocations_long, index=index, columns=assets)
    mean_allocations_df = allocations_df.groupby(
        "period"
    ).mean()  # Mean allocations by period

    # Calculate overall analysis metrics from portfolio performance
    analysis_metrics = calculate_analysis_metrics(portfolio_df["Terminal Wealth"])

    logger.info(
        f"{s_points} out of {s_points} scenarios has now been made. We saw a total of {default_count} "
        f"scenarios where the risk budget glide path strategy defaulted over the period of {p_points} years."
    )
    return portfolio_df, mean_allocations_df, analysis_metrics
