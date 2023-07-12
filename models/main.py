import numpy as np
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from typing import Tuple, Union
from models.dataAnalyser import mean_an_returns, final_stats
from models.MST import minimum_spanning_tree
from models.Clustering import cluster, pick_cluster
from models.ScenarioGeneration import ScenarioGenerator
from models.CVaRtargets import get_cvar_targets
from models.CVaRmodel import cvar_model
from financial_data.etf_isins import ETFlist
from pathlib import Path

pio.renderers.default = "browser"
ROOT_DIR = Path(__file__).parent.parent
# Load our data
weekly_returns = pd.read_parquet(os.path.join(ROOT_DIR, 'financial_data/all_etfs_rets.parquet.gzip'))
tickers = [pair[0] for pair in weekly_returns.columns.values]
names = [pair[1] for pair in weekly_returns.columns.values]


class TradeBot(object):
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

        weekly_returns.columns = tickers

    @staticmethod
    def __plot_backtest(
        performance: pd.DataFrame,
        performance_benchmark: pd.DataFrame,
        composition: pd.DataFrame,
        names: list,
        tickers: list
    ) -> Tuple[px.line, go.Figure]:
        """ METHOD TO PLOT THE BACKTEST RESULTS """

        performance.index = pd.to_datetime(performance.index.values, utc=True)

        # ** PERFORMANCE GRAPH **
        try:
            df_to_plot = pd.concat([performance, performance_benchmark], axis=1)
        except:
            performance.index = [date.date() for date in performance.index]   # needed for old data
            df_to_plot = pd.concat([performance, performance_benchmark], axis=1)

        color_discrete_map = {'Portfolio_Value': '#21304f', 'Benchmark_Value': '#f58f02'}
        fig = px.line(df_to_plot, x=df_to_plot.index, y=df_to_plot.columns,
                      title='Comparison of different strategies', color_discrete_map=color_discrete_map)
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
        composition_color = (px.colors.sequential.turbid
                             + px.colors.sequential.Brwnyl
                             + px.colors.sequential.YlOrBr
                             + px.colors.sequential.gray)
        for isin in composition.columns:
            trace = go.Bar(
                x=composition.index,
                y=composition[isin],
                name=str(isin),
                marker_color=composition_color[idx_color]  # custom color
            )
            data.append(trace)
            idx_color += 1

        layout = go.Layout(barmode='stack')
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(
            title="Portfolio Composition",
            xaxis_title="Number of the Investment Period",
            yaxis_title="Composition",
            legend_title="Name of the Fund")
        fig.layout.yaxis.tickformat = ',.1%'
        fig_composition = fig

        # Show figure if needed
        # fig.show()

        return fig_performance, fig_composition

    def get_stat(self, start_date: str, end_date: str) -> pd.DataFrame:
        """ METHOD COMPUTING ANNUAL RETURNS, ANNUAL STD. DEV. & SHARPE RATIO OF ASSETS """

        # ANALYZE THE DATA for a given time period
        weekly_data = self.weeklyReturns[(self.weeklyReturns.index >= start_date)
                                         & (self.weeklyReturns.index <= end_date)].copy()

        # Create table with summary statistics
        mu_ga = mean_an_returns(weekly_data)  # Annualised geometric mean of returns
        std_dev_a = weekly_data.std(axis=0) * np.sqrt(52)  # Annualised standard deviation of returns
        sharpe = round(mu_ga / std_dev_a, 2)  # Sharpe ratio of each financial product

        # Write all results into a data frame
        stat_df = pd.concat([mu_ga, std_dev_a, sharpe], axis=1)
        stat_df.columns = ["Average Annual Returns", "Standard Deviation of Returns", "Sharpe Ratio"]
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
        ml: str = '',
        ml_subset: Union[list, pd.DataFrame] = None,
        fund_set: list = [],
        optimal_portfolio: list = [],
        benchmark: list = []
    ) -> px.scatter:
        """ METHOD TO PLOT THE OVERVIEW OF THE FINANCIAL PRODUCTS IN TERMS OF RISK AND RETURNS """

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
            data.loc[self.tickers[isin_idx], "Type"] = str(data.loc[self.tickers[isin_idx], "Name"])
            data.loc[self.tickers[isin_idx], "Size"] = 3

        # PLOTTING Data
        color_discrete_map = {'ETF': '#21304f', 'Mutual Fund': '#f58f02',
                              'Funds': '#21304f', "MST subset": '#f58f02',
                              'Cluster 1': '#21304f', 'Cluster 2': '#f58f02',
                              'Benchmark Portfolio': '#f58f02', 'Optimal Portfolio': 'olive'}
        fig = px.scatter(data,
                         x="Standard Deviation of Returns",
                         y="Average Annual Returns",
                         color="Type",
                         size="Size",
                         size_max=8,
                         hover_name="Name",
                         hover_data={"Sharpe Ratio": True, "ISIN": True, "Size": False},
                         color_discrete_map=color_discrete_map,
                         title="Annual Returns and Standard Deviation of Returns from "
                               + start_date[:10] + " to " + end_date[:10]
                         )

        # AXIS IN PERCENTAGES
        fig.layout.yaxis.tickformat = ',.1%'
        fig.layout.xaxis.tickformat = ',.1%'

        # RISK LEVEL MARKER
        min_risk = data['Standard Deviation of Returns'].min()
        max_risk = data['Standard Deviation of Returns'].max()
        risk_level = {"Risk Class 1": 0.005,
                      "Risk Class 2": 0.02,
                      "Risk Class 3": 0.05,
                      "Risk Class 4": 0.10,
                      "Risk Class 5": 0.15,
                      "Risk Class 6": 0.25,
                      "Risk Class 7": max_risk}
        # Initialize dynamic risk levels
        actual_risk_level = set()  
        for i in range(1, 8):
            k = "Risk Class " + str(i)
            if (risk_level[k] >= min_risk) and (risk_level[k] <= max_risk):
                actual_risk_level.add(i)
                
        if max(actual_risk_level) < 7:
            actual_risk_level.add(max(actual_risk_level) + 1)  # Add the final risk level  
            
        for level in actual_risk_level:
            k = "Risk Class " + str(level)
            fig.add_vline(x=risk_level[k], line_width=1, line_dash="dash",
                          line_color="#7c90a0")  # annotation_text=k, annotation_position="top left")
            fig.add_annotation(x=risk_level[k] - 0.01, y=max(data["Average Annual Returns"]), text=k, textangle=-90,
                               showarrow=False)

        # RETURN LEVEL MARKER
        fig.add_hline(y=0, line_width=1.5, line_color="rgba(233, 30, 99, 0.5)")

        # TITLES
        fig.update_annotations(font_color="#000000")
        fig.update_layout(
            xaxis_title="Annualised standard deviation of returns (Risk)",
            yaxis_title="Annualised average returns",
        )
        # Position of legend
        fig.update_layout(legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01
        ))

        return fig

    def mst(
        self, start_date: str, end_date: str, n_mst_runs: int, plot: bool = False
    ) -> Tuple[Union[None, px.scatter], list]:
        """ METHOD TO RUN MST METHOD AND PRINT RESULTS """
        fig, subset_mst = None, []
        
        # Starting subset of data for MST
        subset_mst_df = self.weeklyReturns[(self.weeklyReturns.index >= start_date)
                                           & (self.weeklyReturns.index <= end_date)].copy()

        for i in range(n_mst_runs):
            subset_mst, subset_mst_df, corr_mst_avg, pdi_mst = minimum_spanning_tree(subset_mst_df)

        # PLOTTING RESULTS
        if plot and len(subset_mst) > 0:
            end_df_date = str(subset_mst_df.index.date[-1])
            fig = self.plot_dots(start_date=start_date, end_date=end_df_date, ml="MST", ml_subset=subset_mst)

        return fig, subset_mst

    def clustering(
        self, start_date: str, end_date: str, n_clusters: int, n_assets: int, plot: bool = False
    ) -> Tuple[Union[None, px.scatter], list]:
        """
        METHOD TO RUN MST METHOD AND PRINT RESULTS
        """
        fig = None
        dataset = self.weeklyReturns[(self.weeklyReturns.index >= start_date)
                                     & (self.weeklyReturns.index <= end_date)].copy()
        # CLUSTER DATA
        clusters = cluster(dataset, n_clusters)

        # SELECT ASSETS
        end_dataset_date = str(dataset.index.date[-1])
        clustering_stats = self.get_stat(start_date, end_dataset_date)
        subset_clustering, subset_clustering_df = pick_cluster(data=dataset,
                                                               stat=clustering_stats,
                                                               ml=clusters,
                                                               n_assets=n_assets)  # Number of assets from each cluster

        # PLOTTING DATA
        if plot:
            fig = self.plot_dots(start_date=start_date, end_date=end_dataset_date, ml="Clustering", ml_subset=clusters)

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
        solver: str = "ECOS",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, px.line, go.Figure]:
        """ METHOD TO COMPUTE THE BACKTEST """

        # Find Benchmarks' ISIN codes
        benchmark_isin = [self.tickers[list(self.names).index(name)] for name in benchmarks]

        # Get train and testing datasets
        whole_dataset = self.weeklyReturns[(self.weeklyReturns.index >= start_train_date)
                                           & (self.weeklyReturns.index <= end_test_date)].copy()
        test_dataset = self.weeklyReturns[(self.weeklyReturns.index > start_test_date)
                                          & (self.weeklyReturns.index <= end_test_date)].copy()

        # Create scenario generator
        sg = ScenarioGenerator(np.random.default_rng())

        # SCENARIO GENERATION
        # ---------------------------------------------------------------------------------------------------
        if scenarios_type == 'MonteCarlo':
            scenarios = sg.monte_carlo(data=whole_dataset[subset_of_assets],    # subsetMST_df or subsetCLUST_df
                                       n_simulations=n_simulations,
                                       n_test=len(test_dataset.index))
        else:
            scenarios = sg.bootstrapping(data=whole_dataset[subset_of_assets],  # subsetMST or subsetCLUST
                                         n_simulations=n_simulations,  # number of scenarios per period
                                         n_test=len(test_dataset.index))  # number of periods

        # TARGETS GENERATION
        # ---------------------------------------------------------------------------------------------------
        start_of_test_dataset = str(test_dataset.index.date[0])
        targets, benchmark_port_val = get_cvar_targets(test_date=start_of_test_dataset,
                                                       benchmark=benchmark_isin,  # MSCI World benchmark
                                                       budget=100,
                                                       cvar_alpha=0.05,
                                                       data=whole_dataset,
                                                       scgen=sg,
                                                       n_simulations=n_simulations)
 
        # MATHEMATICAL MODELING
        # ------------------------------------------------------------------
        port_allocation, port_value, port_cvar = cvar_model(test_ret=test_dataset[subset_of_assets],
                                                            scenarios=scenarios,  # Scenarios
                                                            targets=targets,  # Target
                                                            budget=100,
                                                            cvar_alpha=0.05,
                                                            trans_cost=0.001,
                                                            max_weight=1,
                                                            solver=solver)
        #                                                   inaccurate=inaccurate_solution)

        # PLOTTING
        # ------------------------------------------------------------------
        fig_performance, fig_composition = self.__plot_backtest(performance=port_value.copy(),
                                                                performance_benchmark=benchmark_port_val.copy(),
                                                                composition=port_allocation,
                                                                names=self.names,
                                                                tickers=self.tickers)

        # RETURN STATISTICS
        # ------------------------------------------------------------------
        optimal_portfolio_stat = final_stats(port_value)
        benchmark_stat = final_stats(benchmark_port_val)

        return optimal_portfolio_stat, benchmark_stat, fig_performance, fig_composition


if __name__ == "__main__":
    # INITIALIZATION OF THE CLASS
    algo = TradeBot()

    # PLOT INTERACTIVE GRAPH
    algo.plot_dots(start_date="2018-09-24", end_date="2019-09-01")

    # RUN THE MINIMUM SPANNING TREE METHOD
    _, mst_subset_of_assets = algo.mst(start_date="2015-12-23",
                                       end_date="2017-07-01",
                                       n_mst_runs=3,
                                       plot=True)

    # RUN THE CLUSTERING METHOD
    _, clustering_subset_of_assets = algo.clustering(start_date="2015-12-23",
                                                     end_date="2017-07-01",
                                                     n_clusters=3,
                                                     n_assets=10,
                                                     plot=True)

    # RUN THE BACKTEST
    results = algo.backtest(start_train_date="2015-12-23",
                            start_test_date="2018-09-24",
                            end_test_date="2019-09-01",
                            subset_of_assets=mst_subset_of_assets,
                            benchmarks=['BankInvest Danske Aktier W'],
                            scenarios_type='Bootstrapping',
                            n_simulations=500)
