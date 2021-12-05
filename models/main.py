"""
Created on Wed Nov 11 15:50:42 2020

@author: Petr Vanek
"""

from models.dataAnalyser import meanRetAn, finalStat
from models.MST import MinimumSpanningTree
from models.Clustering import Cluster, pickCluster
from models.ScenarioGeneration import MC, BOOT
from models.CVaRtargets import targetsCVaR
from models.CVaRmodel import modelCVaR
from pandas_datareader import data
from data.ETFlist import ETFlist

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

data = pd.read_parquet('data/algostrata_isin.parquet')
tickers = data.columns.values
data_name = pd.read_parquet('data/algostrata_name.parquet')
names = data_name.columns.values

class TradeBot(object):
    """
    Python class analysing financial products and based on machine learning algorithms and mathematical
    optimization suggesting optimal portfolio of assets.
    """

    # def __init__(self, start, end, assets):
    #     # DOWNLOAD THE ADJUSTED DAILY PRICES FROM YAHOO DATABASE
    #     dailyPrices = data.DataReader(assets, 'yahoo', start, end)["Adj Close"]
    #     ## Extra
    #     # test = dailyPrices
    #     # for k in range(len(test.columns)):
    #     #     for i in range(len(test.index)):
    #     #         if math.isnan(float(test.iloc[i, k])):
    #     #             test.iloc[i, k] = test.iloc[i-1, k]
    #     # dailyPrices=test
    #     # GET WEEKLY RETURNS
    #     # Get prices only for Wednesdays and delete Nan columns
    #     pricesWed = dailyPrices[dailyPrices.index.weekday == 2].dropna(axis=1)
    #     # Get weekly returns
    #     self.weeklyReturns = pricesWed.pct_change().drop(pricesWed.index[0])  # drop first NaN row
    #

    def __init__(self):
        self.weeklyReturns = data
        self.tickers = tickers
        self.names = names

    # METHOD COMPUTING ANNUAL RETURNS, ANNUAL STD. DEV. & SHARPE RATIO OF ASSETS
    def __get_stat(self, start, end):

        # ANALYZE THE DATA for a given time period
        weeklyData = self.weeklyReturns[(self.weeklyReturns.index >= start) & (self.weeklyReturns.index <= end)].copy()

        # Create table with summary statistics
        mu_ga = meanRetAn(weeklyData)                   # Annualised geometric mean of returns
        stdev_a = weeklyData.std(axis=0) * np.sqrt(52)  # Annualised standard deviation of returns
        sharpe = round(mu_ga/stdev_a, 2)                # Sharpe ratio of each financial product

        # Write all results into a data frame
        statDf = pd.concat([mu_ga, stdev_a, sharpe], axis=1)
        statDf.columns = ["Average Annual Returns", "Standard Deviation of Returns", "Sharpe Ratio"]
        statDf["ISIN"] = statDf.index                   # Add names into the table
        statDf["Name"] = self.names

        # IS ETF OR NOT? Set size
        for isin in statDf.index:
            if isin in ETFlist:
                statDf.loc[isin, "Type"] = "ETF"
                statDf.loc[isin, "Size"] = 1
            else:
                statDf.loc[isin, "Type"] = "Mutual Fund"
                statDf.loc[isin, "Size"] = 1

        return statDf

    # METHOD TO PLOT THE BACKTEST RESULTS
    def __plot_backtest(self, performance, performanceBenchmark, composition, names, tickers):
        ## FOR Yahoo
        # performance.index = performance.index.date
        ## FOR Morningstar
        performance.index = pd.to_datetime(performance.index.values, utc=True)

        # ** PERFORMANCE GRAPH **
        df_to_plot = pd.concat([performance, performanceBenchmark], axis=1)
        color_discrete_map = {'Portfolio_Value': '#21304f', 'Benchmark_Value': '#f58f02'}
        fig = px.line(df_to_plot, x=df_to_plot.index, y=df_to_plot.columns,
                      title='Comparison of different strategies', color_discrete_map=color_discrete_map)
        figPerf = fig

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
        composition_color = px.colors.sequential.turbid + px.colors.sequential.Brwnyl \
                            + px.colors.sequential.YlOrBr + px.colors.sequential.gray
        for isin in composition.columns:
            trace = go.Bar(
                x=composition.index,
                y=composition[isin],
                name=str(isin),
                marker_color=composition_color[idx_color] #custome color
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
        #fig.show()
        figComp = fig

        return figPerf, figComp

    # METHOD TO PLOT THE OVERVIEW OF THE FINANCIAL PRODUCTS IN TERMS OF RISK AND RETURNS
    def plot_dots(self, start, end, ML=None, MLsubset=None, fundSet=[]):
        # Get statistics for a given time period
        data = self.__get_stat(start, end)

        # IF WE WANT TO HIGHLIGHT THE SUBSET OF ASSETS BASED ON ML
        if ML == "MST":
            data.loc[:, "Type"] = "Funds"
            for fund in MLsubset:
                data.loc[fund, "Type"] = "MST subset"
        if ML == "Clustering":
            data.loc[:, "Type"] = MLsubset.loc[:, "Cluster"]

        # If selected any fund for comparison
        for fund in fundSet:
            isin_idx = list(self.names).index(fund)
            data.loc[self.tickers[isin_idx], "Type"] = str(data.loc[self.tickers[isin_idx], "Name"])
            data.loc[self.tickers[isin_idx], "Size"] = 3

        # PLOTTING Data
        color_discrete_map = {'Mutual Fund': '#21304f', 'ETF': '#f58f02',
                              'Funds': '#21304f', "MST subset": '#f58f02',
                              'Cluster 1': '#21304f', 'Cluster 2': '#f58f02'}
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
                               + start[:10] + " to " + end[:10]
                        )

        # AXIS IN PERCENTAGES
        fig.layout.yaxis.tickformat = ',.1%'
        fig.layout.xaxis.tickformat = ',.1%'

        # RISK LEVEL MARKER
        minRisk = data['Standard Deviation of Returns'].min()
        maxRisk = data['Standard Deviation of Returns'].max()
        riskLevels = {"Risk Class 1" : 0.005, 
                      "Risk Class 2" : 0.02, 
                      "Risk Class 3" : 0.05, 
                      "Risk Class 4" : 0.10,
                      "Risk Class 5" : 0.15, 
                      "Risk Class 6" : 0.25,
                      "Risk Class 7" : maxRisk}           
        actualRiskLevels = set() # Define dynamic risk levels
        for i in range(1,8):
            k = "Risk Class " + str(i)
            if (riskLevels[k] >= minRisk) and (riskLevels[k] <= maxRisk):
                actualRiskLevels.add(i)
        if max(actualRiskLevels) < 7:
            actualRiskLevels.add(max(actualRiskLevels)+1)  # Add the final risk level       
        for l in actualRiskLevels:
            k = "Risk Class " + str(l)
            fig.add_vline(x=riskLevels[k], line_width=1, line_dash="dash", line_color="#7c90a0")# annotation_text=k, annotation_position="top left")
            fig.add_annotation(x=riskLevels[k]-0.01, y=max(data["Average Annual Returns"]), text=k, textangle=-90, showarrow=False)
        
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

    # METHOD TO PREPARE DATA FOR ML AND BACKTESTING
    def setup_data(self, start, end, train_test, train_ratio=0.5, end_train=None, start_test=None):
        self.start = start
        self.end = end
        self.train_test = train_test
        self.AIdata = self.__get_stat(start, end)

        # Get data for a given time interval
        data = self.weeklyReturns[(self.weeklyReturns.index >= start) & (self.weeklyReturns.index <= end)].copy()

        # IF WE DIVIDE DATASET
        if train_test:
            # # DIVIDE DATA INTO TRAINING AND TESTING PARTS
            # breakPoint = int(np.floor(len(data.index) * train_ratio))
            #
            # # DEFINITION OF TRAINING AND TESTING DATASETS
            # self.trainDataset = data.iloc[0:breakPoint, :]
            # self.testDataset = data.iloc[breakPoint:, :]

            self.trainDataset = data[data.index <= end_train]
            self.testDataset = data[data.index > start_test]

            # Get dates
            self.endTrainDate = str(self.trainDataset.index.date[-1])
            self.startTestDate = str(self.testDataset.index.date[0])

            self.dataPlot = self.__get_stat(start, self.endTrainDate)
            self.lenTest = len(self.testDataset.index)
        else:
            self.trainDataset = data
            self.endTrainDate = str(self.trainDataset.index.date[-1])
            self.dataPlot = self.__get_stat(start, end)
            self.lenTest = 0

    # METHOD TO RUN MST METHOD AND PRINT RESULTS
    def mst(self, nMST, plot):
        # Starting subset of data for MST
        self.subsetMST_df = self.trainDataset
        for i in range(nMST):
            self.subsetMST, self.subsetMST_df, self.corrMST_avg, self.PDI_MST = MinimumSpanningTree(self.subsetMST_df)

        # PLOTTING RESULTS
        if plot:
            fig = self.plot_dots(start=self.start, end=self.endTrainDate, ML="MST", MLsubset=self.subsetMST)
            return fig

    # METHOD TO RUN MST METHOD AND PRINT RESULTS
    def clustering(self, nClusters, nAssets, plot):
        # CLUSTER DATA
        clusters = Cluster(self.trainDataset, nClusters=nClusters, dendogram=False)

        # SELECT ASSETS
        self.subsetCLUST, self.subsetCLUST_df = pickCluster(data=self.trainDataset,
                                                            stat=self.dataPlot,
                                                            ML=clusters,
                                                            nAssets=nAssets)  # Number of assets selected from each cluster

        # PLOTTING DATA
        if plot:
            fig = self.plot_dots(start=self.start, end=self.endTrainDate, ML="Clustering", MLsubset=clusters)
            return fig

    # METHOD TO COMPUTE THE BACKTEST
    def backtest(self, assets, benchmark, scenarios, nSimulations, plot=True):

        # Find Benchmarks' ISIN codes
        benchmark_isin = [self.tickers[list(self.names).index(name)] for name in benchmark]

        # SELECT THE WORKING SUBSET
        if assets == 'MST':
            subset = self.subsetMST
        elif assets == 'Clustering':
            subset = self.subsetCLUST
        else:
            subset = assets

        # SCENERIO GENERATION
        # ---------------------------------------------------------------------------------------------------
        if scenarios == 'MonteCarlo':
            scenarios = MC(data=self.trainDataset.loc[:, self.trainDataset.columns.isin(subset)],                      # subsetMST_df or subsetCLUST_df
                           nSim=nSimulations,
                           N_test=self.lenTest)
        else:
            scenarios = BOOT(data=self.weeklyReturns[subset],   # subsetMST or subsetCLUST
                             nSim=nSimulations,                 # number of scenarios per period
                             N_test=self.lenTest)

        # TARGETS GENERATION
        # ---------------------------------------------------------------------------------------------------
        targets, benchmarkPortVal = targetsCVaR(start_date=self.start,
                                                end_date=self.end,
                                                test_date=self.startTestDate,
                                                benchmark=benchmark_isin,        # MSCI World benchmark
                                                test_index=self.testDataset.index.date,
                                                budget=100,
                                                cvar_alpha=0.05,
                                                data=self.weeklyReturns)

        # MATHEMATICAL MODELING
        # ------------------------------------------------------------------
        portAllocation, portValue, portCVaR = modelCVaR(testRet=self.testDataset[subset],
                                                        scen=scenarios,    # Scenarios
                                                        targets=targets,   # Target
                                                        budget=100,
                                                        cvar_alpha=0.05,
                                                        trans_cost=0.001,
                                                        max_weight=1)
        # PLOTTING
        # ------------------------------------------------------------------
        if plot:
            figPerf, figComp = self.__plot_backtest(performance=portValue.copy(),
                                                    performanceBenchmark=benchmarkPortVal.copy(),
                                                    composition=portAllocation,
                                                    names=self.names,
                                                    tickers=self.tickers)

        # RETURN STATISTICS
        # ------------------------------------------------------------------
        optimal_portfolio_stat = finalStat(portValue)
        benchmark_stat = finalStat(benchmarkPortVal)

        if plot:
            return optimal_portfolio_stat, benchmark_stat, figPerf, figComp
        else:
            return optimal_portfolio_stat, benchmark_stat


if __name__ == "__main__":

    # INITIALIZATION OF THE CLASS
    #algo = TradeBot(start="2011-07-01", end="2021-07-01", assets=tickers)
    algo = TradeBot()

    # PLOT INTERACTIVE GRAPH
    algo.plot_dots(start="2018-09-24", end="2019-09-01")

    # SETUP WORKING DATASET, DIVIDE DATASET INTO TRAINING AND TESTING PART?
    algo.setup_data(start="2015-12-23", end="2018-08-22", train_test=True, train_ratio=0.6)

    # RUN THE MINIMUM SPANNING TREE METHOD
    algo.mst(nMST=3, plot=True)

    # RUN THE CLUSTERING METHOD
    algo.clustering(nClusters=3, nAssets=10, plot=True)

    # RUN THE BACKTEST
    results = algo.backtest(assets='MST',
                            benchmark=['URTH'],
                            scenarios='Bootstrapping',
                            nSimulations=500,
                            plot=True)
