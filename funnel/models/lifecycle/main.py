import os
from itertools import cycle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from loguru import logger
from scipy.stats import gaussian_kde

from funnel.models.lifecycle.benchmark import calculate_target_allocation
from funnel.models.lifecycle.glidePathCreator import RiskCurveGenerator
from funnel.models.lifecycle.MVOlifecycleModel import (
    get_port_allocations,
    riskadjust_model_scen,
)
from funnel.models.lifecycle.ScenarioGeneration import (
    MomentGenerator,
    ScenarioGenerator,
)
from funnel.models.MST import minimum_spanning_tree

pio.renderers.default = "browser"
ROOT_DIR = Path(__file__).parent.parent
# Load our data
weekly_returns = pd.read_parquet(
    os.path.join(ROOT_DIR, "funnel/financial_data/all_etfs_rets.parquet.gzip")
)
ticker_name_df = pd.DataFrame(
    {
        "Ticker": [pair[0] for pair in weekly_returns.columns.values],
        "Name": [pair[1] for pair in weekly_returns.columns.values],
    }
)

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
]


# TODO Simplify... looks like a copy-paste from main
class TradeBot:
    def __init__(self):
        self.tickers = ticker_name_df.Ticker
        self.names = ticker_name_df.Name
        self.weeklyReturns = weekly_returns
        self.min_date = str(weekly_returns.index[0])
        self.max_date = str(weekly_returns.index[-2])
        self.withdrawal_lst = withdrawal_lst

        weekly_returns.columns = ticker_name_df.Ticker

    @staticmethod
    def __plot_portfolio_densities(
        portfolio_performance_dict: dict,
    ) -> Tuple[go.Figure]:
        """METHOD TO PLOT THE BACKTEST RESULTS"""

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

    def mst(
        self, start_date: str, end_date: str, n_mst_runs: int
    ) -> Tuple[list, pd.DataFrame]:
        """METHOD TO RUN MST METHOD AND PRINT RESULTS"""
        subset_mst = []

        # Starting subset of data for MST
        subset_mst_df = self.weeklyReturns[
            (self.weeklyReturns.index >= start_date)
            & (self.weeklyReturns.index <= end_date)
        ].copy()

        for i in range(n_mst_runs):
            subset_mst, subset_mst_df, corr_mst_avg, pdi_mst = minimum_spanning_tree(
                subset_mst_df
            )

        return subset_mst, subset_mst_df

    def scenario_analysis(
        self,
        subset_of_assets: list,
        scenarios_type: str,
        n_simulations: int,
        end_year: int,
        risk_test: str,
        risk_class: list,
        compare_with_naive: bool = True,
        solver: str = "ECOS",
    ) -> Tuple[dict, pd.DataFrame, go.Figure]:
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

        elif scenarios_type == "FHS":
            scenarios = sg.FHS(
                data=self.weeklyReturns[subset_of_assets],
                n_simulations=n_simulations,
                n_years=n_periods,
            )
        else:
            logger.exception(
                "❌ Currently we only simulate returns with the Monte Carlo method or FHS."
            )

        # ------------------------------- Risk Target Generation -------------------------------
        if risk_test == "simpleLinear":
            targets_risk = pd.DataFrame(
                np.linspace(0.2, 0.05, n_periods), columns=["Linear Glide Path"]
            )

        elif risk_test == "investmentFunnel":
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

            targets_risk = generator.generate_curves()
            targets_risk = generator.filter_columns_by_risk_class(
                targets_risk, risk_class
            )

            # ------------------------------- Allocation Target Generation -------------------------------
            allocation_targets = {}
            for r in targets_risk.columns:
                targets = get_port_allocations(
                    mu_lst=mu,
                    sigma_lst=sigma,
                    targets=targets_risk[r],
                    max_weight=1 / 3,
                    solver=solver,
                )
                allocation_targets[f"{r}"] = targets

            # ------------------------------- Generate Naive 1/N in stock/bond portfolio -------------------------------
            if compare_with_naive:
                class_alloc_targets = pd.DataFrame(np.linspace(1, 0, n_periods))
                naive_allocation = calculate_target_allocation(
                    class_alloc_targets, subset_of_assets
                )
                allocation_targets["1/N bond/stock portfolio"] = naive_allocation
        else:
            logger.debug("Currently we only have the Markowitz model approach.")

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
                solver="ECOS",
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
        return terminal_wealth_dict, exhibition_summary, fig_performance


if __name__ == "__main__":
    # INITIALIZATION OF THE CLASS
    algo = TradeBot()
    mst_ISIN_all, mstdf_all = algo.mst(
        start_date="2000-01-01", end_date="2024-01-01", n_mst_runs=4
    )

    # RUN THE BACKTEST
    results = algo.scenario_analysis(
        subset_of_assets=mst_ISIN_all,
        scenarios_type="MonteCarlo",
        n_simulations=1000,
        end_year=2050,
        risk_test="investmentFunnel",
        risk_class=[3, 4, 5, 6, 7],
    )
