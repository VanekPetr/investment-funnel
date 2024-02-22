import os
from itertools import cycle
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from loguru import logger
from scipy.stats import gaussian_kde

from funnel.financial_data.etf_isins import ETFlist
from funnel.models.Clustering import cluster, pick_cluster
from funnel.models.CVaRmodel import cvar_model
from funnel.models.CVaRtargets import get_cvar_targets
from funnel.models.dataAnalyser import final_stats, mean_an_returns
from funnel.models.lifecycle.assetClassGlidePaths import calculate_target_allocation
from funnel.models.lifecycle.glidePathCreator import RiskCurveGenerator
from funnel.models.lifecycle.MVOlifecycleModel import (
    get_port_allocations,
    riskadjust_model_scen,
)
from funnel.models.MST import minimum_spanning_tree
from funnel.models.MVOmodel import mvo_model
from funnel.models.MVOtargets import get_mvo_targets
from funnel.models.ScenarioGeneration import MomentGenerator, ScenarioGenerator

withdrawal_lst = [
    27200,
    33900,
    37200,
    39800,
    42300,
    44500,
    46400,
    48200,
    50000,
    51800,
    53600,
    46100,
    47000,
    48100,
    49200,
    50900,
    52200,
    53500,
    54800,
    56000,
    57200,
    58400,
    65500,
    89900,
    90800,
    91800,
    92800,
    93700,
    94600,
    95500,
    96400,
    97300,
    98200,
    99100,
    100000,
    100900,
    101800,
    102700,
    103600,
    104500,
    105400,
    106300,
    107200,
    108100,
    109000,
    109900,
    110800,
    111700,
    112600,
    113500,
]

pio.renderers.default = "browser"
ROOT_DIR = Path(__file__).parent.parent
# Load our data
weekly_returns = pd.read_parquet(
    os.path.join(ROOT_DIR, "financial_data/all_etfs_rets.parquet.gzip")
)
tickers = [pair[0] for pair in weekly_returns.columns.values]
names = [pair[1] for pair in weekly_returns.columns.values]


class TradeBot:
    """
    Python class analysing financial products and based on machine learning algorithms and mathematical
    optimization suggesting optimal portfolio of assets.
    """

    def __init__(self):
        self.tickers = tickers
        self.names = names
        self.weeklyReturns = weekly_returns
        self.min_date = str(weekly_returns.index[0])
        self.max_date = str(weekly_returns.index[-2])
        self.withdrawal_lst = withdrawal_lst

        weekly_returns.columns = tickers

    @staticmethod
    def __plot_backtest(
        performance: pd.DataFrame,
        performance_benchmark: pd.DataFrame,
        composition: pd.DataFrame,
        names: list,
        tickers: list,
    ) -> Tuple[px.line, go.Figure]:
        """METHOD TO PLOT THE BACKTEST RESULTS"""

        performance.index = pd.to_datetime(performance.index.values, utc=True)

        # ** PERFORMANCE GRAPH **
        try:
            df_to_plot = pd.concat([performance, performance_benchmark], axis=1)
        except Exception:
            logger.warning("⚠️ Old data format.")
            performance.index = [
                date.date() for date in performance.index
            ]  # needed for old data
            df_to_plot = pd.concat([performance, performance_benchmark], axis=1)

        color_discrete_map = {
            "Portfolio_Value": "#21304f",
            "Benchmark_Value": "#f58f02",
        }
        fig = px.line(
            df_to_plot,
            x=df_to_plot.index,
            y=df_to_plot.columns,
            title="Comparison of different strategies",
            color_discrete_map=color_discrete_map,
        )
        fig_performance = fig

        # ** COMPOSITION GRAPH **
        # change ISIN to NAMES in allocation df
        composition_names = []
        for ticker in composition.columns:
            ticker_index = list(tickers).index(ticker)
            composition_names.append(list(names)[ticker_index])
        composition.columns = composition_names

        composition = composition.loc[:, (composition != 0).any(axis=0)]
        data = []
        idx_color = 0
        composition_color = (
            px.colors.sequential.turbid
            + px.colors.sequential.Brwnyl
            + px.colors.sequential.YlOrBr
            + px.colors.sequential.gray
            + px.colors.sequential.Mint
            + px.colors.sequential.dense
            + px.colors.sequential.Plasma
            + px.colors.sequential.Viridis
            + px.colors.sequential.Cividis
        )
        for isin in composition.columns:
            trace = go.Bar(
                x=composition.index,
                y=composition[isin],
                name=str(isin),
                marker_color=composition_color[idx_color],  # custom color
            )
            data.append(trace)
            idx_color += 1

        layout = go.Layout(barmode="stack")
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(
            title="Portfolio Composition",
            xaxis_title="Number of the Investment Period",
            yaxis_title="Composition",
            legend_title="Name of the Fund",
        )
        fig.layout.yaxis.tickformat = ",.1%"
        fig_composition = fig

        # Show figure if needed
        # fig.show()

        return fig_performance, fig_composition

    @staticmethod
    def __plot_portfolio_densities(
        portfolio_performance_dict: dict,
    ) -> Tuple[go.Figure]:
        """METHOD TO PLOT THE LIFECYCLE SIMULATION RESULTS"""

        # Define colors
        colors = [
            "#4099da",  # blue
            "#b7ada5",  # secondary
            "#8ecdc8",  # aqua
            "#e85757",  # coral
            "#fdd779",  # sun
            "#644c76",  # eggplant
            "#3b4956",  # dark
            "#99A4AE",  # gray50
            "#3B4956",  # gray100
            "#D8D1CA",  # warmGray50
            "#B7ADA5",  # warmGray100
            "#FFFFFF",  # white
        ]

        color_cycle = cycle(colors)  # To cycle through colors
        fig = go.Figure()

        for label, df in portfolio_performance_dict.items():
            # Kernel Density Estimation for each dataset
            kde = gaussian_kde(df["Terminal Wealth"])

            # Generating a range of values to evaluate the KDE
            x_min = df["Terminal Wealth"].min()
            x_max = df["Terminal Wealth"].max()
            x = np.linspace(x_min, x_max, 1000)

            # Evaluate the KDE
            density = kde(x)

            # Create line plot trace for this dataset
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=density,
                    mode="lines",
                    name=label,  # Use the dictionary key as the label
                    line=dict(
                        color=next(color_cycle)
                    ),  # Assign color from Orsted-Colors
                )
            )

        # Add a dashed vertical line at x=0
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=0,
            y1=max([max(density) for _, df in portfolio_performance_dict.items()]),
            # Adjust y1 to the max density for better visibility
            line=dict(
                color="Black",
                width=2,
                dash="dash",  # Define dash pattern
            ),
        )
        """
        # Update the layout
        fig.update_layout(
            title_text='Density function(s) of terminal wealth for risk classes in 1000 different scenarios.',
            xaxis_title='Terminal Wealth',
            yaxis_title='Density',
            legend_title='Risk Class',
            template='plotly_white'
        )
        """
        # Update the layout with larger fonts
        fig.update_layout(
            title_text="Density function(s) of terminal wealth for risk classes in 1000 different scenarios.",
            title_font=dict(size=24),  # Increase title font size
            xaxis_title="Terminal Wealth",
            xaxis_title_font=dict(size=18),  # Increase x-axis title font size
            xaxis_tickfont=dict(size=16),  # Increase x-axis tick label font size
            yaxis_title="Density",
            yaxis_title_font=dict(size=18),  # Increase y-axis title font size
            yaxis_tickfont=dict(size=16),  # Increase y-axis tick label font size
            legend_title="Risk Class",
            legend_title_font=dict(size=18),  # Increase legend title font size
            legend_font=dict(size=16),  # Increase legend text font size
            template="plotly_white",
        )

        # Show the figure in a browser
        fig.show(renderer="browser")

        return fig

    def get_stat(self, start_date: str, end_date: str) -> pd.DataFrame:
        """METHOD COMPUTING ANNUAL RETURNS, ANNUAL STD. DEV. & SHARPE RATIO OF ASSETS"""

        # ANALYZE THE DATA for a given time period
        weekly_data = self.weeklyReturns[
            (self.weeklyReturns.index >= start_date)
            & (self.weeklyReturns.index <= end_date)
        ].copy()

        # Create table with summary statistics
        mu_ga = mean_an_returns(weekly_data)  # Annualised geometric mean of returns
        std_dev_a = weekly_data.std(axis=0) * np.sqrt(
            52
        )  # Annualised standard deviation of returns
        sharpe = round(mu_ga / std_dev_a, 2)  # Sharpe ratio of each financial product

        # Write all results into a data frame
        stat_df = pd.concat([mu_ga, std_dev_a, sharpe], axis=1)
        stat_df.columns = [
            "Average Annual Returns",
            "Standard Deviation of Returns",
            "Sharpe Ratio",
        ]
        stat_df["ISIN"] = stat_df.index  # Add names into the table
        stat_df["Name"] = self.names

        # IS ETF OR NOT? Set size
        for isin in stat_df.index:
            if isin in ETFlist:
                stat_df.loc[isin, "Type"] = "ETF"
                stat_df.loc[isin, "Size"] = 1
            else:
                stat_df.loc[isin, "Type"] = "ETF"
                stat_df.loc[isin, "Size"] = 1

        return stat_df

    def plot_dots(
        self,
        start_date: str,
        end_date: str,
        ml: str = "",
        ml_subset: Union[list, pd.DataFrame] = None,
        fund_set: list = [],
        optimal_portfolio: list = [],
        benchmark: list = [],
    ) -> px.scatter:
        """METHOD TO PLOT THE OVERVIEW OF THE FINANCIAL PRODUCTS IN TERMS OF RISK AND RETURNS"""

        # Get statistics for a given time period
        data = self.get_stat(start_date, end_date)

        # Add data about the optimal portfolio and benchmark for plotting
        if len(optimal_portfolio) > 0:
            data.loc[optimal_portfolio[4]] = optimal_portfolio
        if len(benchmark) > 0:
            data.loc[benchmark[4]] = benchmark

        # IF WE WANT TO HIGHLIGHT THE SUBSET OF ASSETS BASED ON ML
        if ml == "MST":
            data.loc[:, "Type"] = "Funds"
            for fund in ml_subset:
                data.loc[fund, "Type"] = "MST subset"
        if ml == "Clustering":
            data.loc[:, "Type"] = ml_subset.loc[:, "Cluster"]

        # If selected any fund for comparison
        for fund in fund_set:
            isin_idx = list(self.names).index(fund)
            data.loc[self.tickers[isin_idx], "Type"] = str(
                data.loc[self.tickers[isin_idx], "Name"]
            )
            data.loc[self.tickers[isin_idx], "Size"] = 3

        # PLOTTING Data
        color_discrete_map = {
            "ETF": "#21304f",
            "Mutual Fund": "#f58f02",
            "Funds": "#21304f",
            "MST subset": "#f58f02",
            "Cluster 1": "#21304f",
            "Cluster 2": "#f58f02",
            "Benchmark Portfolio": "#f58f02",
            "Optimal Portfolio": "olive",
        }
        fig = px.scatter(
            data,
            x="Standard Deviation of Returns",
            y="Average Annual Returns",
            color="Type",
            size="Size",
            size_max=8,
            hover_name="Name",
            hover_data={"Sharpe Ratio": True, "ISIN": True, "Size": False},
            color_discrete_map=color_discrete_map,
            title="Annual Returns and Standard Deviation of Returns from "
            + start_date[:10]
            + " to "
            + end_date[:10],
        )

        # AXIS IN PERCENTAGES
        fig.layout.yaxis.tickformat = ",.1%"
        fig.layout.xaxis.tickformat = ",.1%"

        # RISK LEVEL MARKER
        min_risk = data["Standard Deviation of Returns"].min()
        max_risk = data["Standard Deviation of Returns"].max()
        risk_level = {
            "Risk Class 1": 0.005,
            "Risk Class 2": 0.02,
            "Risk Class 3": 0.05,
            "Risk Class 4": 0.10,
            "Risk Class 5": 0.15,
            "Risk Class 6": 0.25,
            "Risk Class 7": max_risk,
        }
        # Initialize dynamic risk levels
        actual_risk_level = set()
        for i in range(1, 8):
            k = "Risk Class " + str(i)
            if (risk_level[k] >= min_risk) and (risk_level[k] <= max_risk):
                actual_risk_level.add(i)

        if max(actual_risk_level) < 7:
            actual_risk_level.add(
                max(actual_risk_level) + 1
            )  # Add the final risk level

        for level in actual_risk_level:
            k = "Risk Class " + str(level)
            fig.add_vline(
                x=risk_level[k], line_width=1, line_dash="dash", line_color="#7c90a0"
            )  # annotation_text=k, annotation_position="top left")
            fig.add_annotation(
                x=risk_level[k] - 0.01,
                y=max(data["Average Annual Returns"]),
                text=k,
                textangle=-90,
                showarrow=False,
            )

        # RETURN LEVEL MARKER
        fig.add_hline(y=0, line_width=1.5, line_color="rgba(233, 30, 99, 0.5)")

        # TITLES
        fig.update_annotations(font_color="#000000")
        fig.update_layout(
            xaxis_title="Annualised standard deviation of returns (Risk)",
            yaxis_title="Annualised average returns",
        )
        # Position of legend
        fig.update_layout(legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01))

        return fig

    def mst(
        self, start_date: str, end_date: str, n_mst_runs: int, plot: bool = False
    ) -> Tuple[Union[None, px.scatter], list]:
        """METHOD TO RUN MST METHOD AND PRINT RESULTS"""
        fig, subset_mst = None, []

        # Starting subset of data for MST
        subset_mst_df = self.weeklyReturns[
            (self.weeklyReturns.index >= start_date)
            & (self.weeklyReturns.index <= end_date)
        ].copy()

        for i in range(n_mst_runs):
            subset_mst, subset_mst_df, corr_mst_avg, pdi_mst = minimum_spanning_tree(
                subset_mst_df
            )

        # PLOTTING RESULTS
        if plot and len(subset_mst) > 0:
            end_df_date = str(subset_mst_df.index.date[-1])
            fig = self.plot_dots(
                start_date=start_date,
                end_date=end_df_date,
                ml="MST",
                ml_subset=subset_mst,
            )

        return fig, subset_mst

    def clustering(
        self,
        start_date: str,
        end_date: str,
        n_clusters: int,
        n_assets: int,
        plot: bool = False,
    ) -> Tuple[Union[None, px.scatter], list]:
        """
        METHOD TO RUN MST METHOD AND PRINT RESULTS
        """
        fig = None
        dataset = self.weeklyReturns[
            (self.weeklyReturns.index >= start_date)
            & (self.weeklyReturns.index <= end_date)
        ].copy()
        # CLUSTER DATA
        clusters = cluster(dataset, n_clusters)

        # SELECT ASSETS
        end_dataset_date = str(dataset.index.date[-1])
        clustering_stats = self.get_stat(start_date, end_dataset_date)
        subset_clustering, subset_clustering_df = pick_cluster(
            data=dataset, stat=clustering_stats, ml=clusters, n_assets=n_assets
        )  # Number of assets from each cluster

        # PLOTTING DATA
        if plot:
            fig = self.plot_dots(
                start_date=start_date,
                end_date=end_dataset_date,
                ml="Clustering",
                ml_subset=clusters,
            )

        return fig, subset_clustering

    def backtest(
        self,
        start_train_date: str,
        start_test_date: str,
        end_test_date: str,
        subset_of_assets: list,
        benchmarks: list,
        scenarios_type: str,
        n_simulations: int,
        model: str,
        solver: str = "ECOS",
        lower_bound: int = 0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, px.line, go.Figure]:
        """METHOD TO COMPUTE THE BACKTEST"""

        # Find Benchmarks' ISIN codes
        benchmark_isin = [
            self.tickers[list(self.names).index(name)] for name in benchmarks
        ]

        # Get train and testing datasets
        whole_dataset = self.weeklyReturns[
            (self.weeklyReturns.index >= start_train_date)
            & (self.weeklyReturns.index <= end_test_date)
        ].copy()
        test_dataset = self.weeklyReturns[
            (self.weeklyReturns.index > start_test_date)
            & (self.weeklyReturns.index <= end_test_date)
        ].copy()

        # SCENARIO GENERATION
        # ---------------------------------------------------------------------------------------------------
        # Create scenario generator
        sg = ScenarioGenerator(np.random.default_rng())

        if model == "Markowitz model" or scenarios_type == "MonteCarlo":
            sigma_lst, mu_lst = MomentGenerator.generate_sigma_mu_for_test_periods(
                data=whole_dataset[subset_of_assets], n_test=len(test_dataset.index)
            )

        if scenarios_type == "MonteCarlo":
            scenarios = sg.monte_carlo(
                data=whole_dataset[subset_of_assets],  # subsetMST_df or subsetCLUST_df
                n_simulations=n_simulations,
                n_test=len(test_dataset.index),
                sigma_lst=sigma_lst,
                mu_lst=mu_lst,
            )
        else:
            scenarios = sg.bootstrapping(
                data=whole_dataset[subset_of_assets],  # subsetMST or subsetCLUST
                n_simulations=n_simulations,  # number of scenarios per period
                n_test=len(test_dataset.index),
            )  # number of periods

        # TARGETS GENERATION
        # ---------------------------------------------------------------------------------------------------
        start_of_test_dataset = str(test_dataset.index.date[0])
        if model == "Markowitz model":
            targets, benchmark_port_val = get_mvo_targets(
                test_date=start_of_test_dataset,
                benchmark=benchmark_isin,
                budget=100,
                data=whole_dataset,
            )

        else:
            targets, benchmark_port_val = get_cvar_targets(
                test_date=start_of_test_dataset,
                benchmark=benchmark_isin,
                budget=100,
                cvar_alpha=0.05,
                data=whole_dataset,
                scgen=sg,
                n_simulations=n_simulations,
            )

        # MATHEMATICAL MODELING
        # ---------------------------------------------------------------------------------------------------
        if model == "Markowitz model":
            port_allocation, port_value, port_cvar = mvo_model(
                test_ret=test_dataset[subset_of_assets],
                mu_lst=mu_lst,
                sigma_lst=sigma_lst,
                targets=targets,
                budget=100,
                trans_cost=0.001,
                max_weight=1,
                solver=solver,
                lower_bound=lower_bound,
            )
        #                                                      inaccurate=inaccurate_solution)

        else:
            port_allocation, port_value, port_cvar = cvar_model(
                test_ret=test_dataset[subset_of_assets],
                scenarios=scenarios,  # Scenarios
                targets=targets,  # Target
                budget=100,
                cvar_alpha=0.05,
                trans_cost=0.001,
                max_weight=1,
                solver=solver,
                lower_bound=lower_bound,
            )
        #                                                       inaccurate=inaccurate_solution)

        # PLOTTING
        # ------------------------------------------------------------------
        fig_performance, fig_composition = self.__plot_backtest(
            performance=port_value.copy(),
            performance_benchmark=benchmark_port_val.copy(),
            composition=port_allocation,
            names=self.names,
            tickers=self.tickers,
        )

        # RETURN STATISTICS
        # ------------------------------------------------------------------
        optimal_portfolio_stat = final_stats(port_value)
        benchmark_stat = final_stats(benchmark_port_val)

        return optimal_portfolio_stat, benchmark_stat, fig_performance, fig_composition

    def scenario_analysis(
        self,
        subset_of_assets: list,
        scenarios_type: str,
        n_simulations: int,
        end_year: int,
        risk_test: str,
        risk_class: list,
        compare_with_naive: bool = False,
    ) -> Tuple[dict, pd.DataFrame, go.Figure, go.Figure]:
        """METHOD TO COMPUTE THE LIFECYCLE SCENARIO ANALYSIS"""

        # ------------------------------- SCENARIO GENERATION -------------------------------
        n_periods = end_year - 2023
        sg = ScenarioGenerator(np.random.default_rng())

        if scenarios_type == "MonteCarlo":
            sigma, mu, sigma_weekly, mu_weekly = (
                MomentGenerator.generate_annual_sigma_mu_with_risk_free(
                    data=self.weeklyReturns[subset_of_assets]
                )
            )
            scenarios = sg.MC_simulation_annual_from_weekly(
                weekly_mu=mu_weekly,
                weekly_sigma=sigma_weekly,
                n_simulations=n_simulations,
                n_years=n_periods,
            )
        else:
            logger.debug(
                "We are currently only able to make scenario analysis based on Monte Carlo simulation."
            )

        # ------------------------------- Risk Target Generation -------------------------------
        # TODO this is not in use and should be removed
        # if risk_test == "simpleLinear":
        #     targets_risk = pd.DataFrame(
        #         np.linspace(0.2, 0.05, n_periods), columns=["Linear Glide Path"]
        #     )

        if risk_test == "investmentFunnel":
            # Configuration and usage
            generator = RiskCurveGenerator(
                periods=n_periods,
                risk_floor=0.02,
                std_devs={
                    "Risk Class 3": 0.05,
                    "Risk Class 4": 0.10,
                    "Risk Class 5": 0.15,
                    "Risk Class 6": 0.25,
                    "Risk Class 7": 0.30,
                },
            )

            glide_paths_df, fig_glidepaths = generator.generate_curves()
            glide_paths_df = generator.filter_columns_by_risk_class(
                glide_paths_df, risk_class
            )

        # ------------------------------- Allocation Target Generation -------------------------------
        allocation_targets = {}
        for r in glide_paths_df.columns:
            targets = get_port_allocations(
                mu_lst=mu,
                sigma_lst=sigma,
                targets=glide_paths_df[r],
                max_weight=1 / 3,
                solver="ECOS_BB",
            )
            allocation_targets[f"{r}"] = targets

        # ------------------------------- Generate Naive 1/N in stock/bond portfolio -------------------------------
        if compare_with_naive:
            # TODO: this fails for 2030 and risk_classes [3]
            class_alloc_targets = pd.DataFrame(np.linspace(1, 0, n_periods))
            naive_allocation = calculate_target_allocation(
                class_alloc_targets, subset_of_assets
            )
            allocation_targets["1/N bond/stock portfolio"] = naive_allocation

        # ------------------------------- MATHEMATICAL MODELING -------------------------------
        exhibition_summary = pd.DataFrame()
        terminal_wealth_dict = {}

        for key, df in allocation_targets.items():
            logger.info(
                f"Optimizing portfolio for {key} over {n_simulations} scenarios. An info message will "
                f"appear, when we are halfway through the scenarios for the current strategy."
            )
            portfolio_df, mean_allocations_df, analysis_metrics = riskadjust_model_scen(
                scen=scenarios[:, :, :],
                targets=df,
                budget=837000,
                trans_cost=0.002,
                withdrawal_lst=self.withdrawal_lst,
                interest_rate=0.04,
            )

            # Add the analysis_metrics DataFrame as a new column in the storage DataFrame
            exhibition_summary[key] = analysis_metrics.squeeze()

            portfolio_df["Terminal Wealth"] = pd.to_numeric(
                portfolio_df["Terminal Wealth"], errors="coerce"
            )
            terminal_wealth_dict[f"{key}"] = portfolio_df

        # ------------------------------- PLOTTING -------------------------------
        fig_performance = self.__plot_portfolio_densities(
            portfolio_performance_dict=terminal_wealth_dict
        )

        # ------------------------------- RETURN STATISTICS -------------------------------
        return terminal_wealth_dict, exhibition_summary, fig_performance, fig_glidepaths


if __name__ == "__main__":
    # INITIALIZATION OF THE CLASS
    algo = TradeBot()

    # PLOT INTERACTIVE GRAPH
    algo.plot_dots(start_date="2018-09-24", end_date="2019-09-01")

    # RUN THE MINIMUM SPANNING TREE METHOD
    _, mst_subset_of_assets = algo.mst(
        start_date="2000-01-01", end_date="2024-01-01", n_mst_runs=5, plot=False
    )

    # RUN THE CLUSTERING METHOD
    _, clustering_subset_of_assets = algo.clustering(
        start_date="2015-12-23",
        end_date="2017-07-01",
        n_clusters=3,
        n_assets=10,
        plot=True,
    )

    # RUN THE LIFECYCLE
    lifecycle = algo.scenario_analysis(
        subset_of_assets=mst_subset_of_assets,
        scenarios_type="MonteCarlo",
        n_simulations=1000,
        end_year=2050,
        risk_test="investmentFunnel",
        risk_class=[3, 4, 5, 6, 7],
    )

    # RUN THE BACKTEST
    backtest = algo.backtest(
        start_train_date="2015-12-23",
        start_test_date="2018-09-24",
        end_test_date="2019-09-01",
        subset_of_assets=mst_subset_of_assets,
        benchmarks=["BankInvest Danske Aktier W"],
        scenarios_type="Bootstrapping",
        n_simulations=500,
        model="Markowitz model",
        lower_bound=0,
    )
