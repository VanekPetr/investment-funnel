from typing import NamedTuple

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import dcc, html

from .components_and_styles.ai_feature_selection_page import divs as ai_feature_selection_divs
from .components_and_styles.backtest_page import divs as backtest_divs
from .components_and_styles.general import mobile_page, sideBar
from .components_and_styles.lifecycle_page import divs as lifecycle_divs
from .components_and_styles.market_overview_page import divs as market_overview_divs

Layout = NamedTuple('Layout', [
    ('page_1', html.Div),
    ('page_2', html.Div),
    ('page_3', html.Div),
    ('page_4', html.Div),
    ('page_mobile', html.Div),
])

def divs(algo) -> Layout:
    overview = market_overview_divs(algo)
    lifecycle = lifecycle_divs(algo)
    backtest = backtest_divs(algo)
    ai_feature_selection = ai_feature_selection_divs(algo)

    # *** LIFECYCLE ***
    page_4_layout = html.Div(
        [
            # Row 1 - body
            dbc.Row(
                [
                    # Row 1, Col 1 - navigation bar
                    dbc.Col([sideBar]),
                    # Row 1, col 2 - text description
                    dbc.Col([lifecycle.options]),
                    # Row 1, Col 3 - table
                    dbc.Col([lifecycle.results, lifecycle.spinner]),
                ]
            )
        ]
    )

    # *** BACK-TESTING ***
    page_3_layout = html.Div(
        [
            # Row 1 - body
            dbc.Row(
                [
                    # Row 1, Col 1 - navigation bar
                    dbc.Col([sideBar]),
                    # Row 1, col 2 - text description
                    dbc.Col([backtest.options]),
                    # Row 1, Col 3 - table
                    dbc.Col([backtest.results, backtest.spinner]),
                ]
            )
        ]
    )


    # *** AI Feature Selection ***
    page_2_layout = html.Div(
        [
            # Row 1 - body
            dbc.Row(
                [
                    # Row 1, Col 1 - navigation bar
                    dbc.Col([sideBar]),
                    # Row 1, col 2 - set-up
                    dbc.Col([ai_feature_selection.options]),
                    # Row 1, Col 3 - table
                    dbc.Col([ai_feature_selection.graph, ai_feature_selection.spinner]),
                ]
            )
        ]
    )

    # *** MARKET OVERVIEW ***
    page_1_layout = html.Div(
        [
            # Row 1 - body
            dbc.Row(
                [
                    # Row 1, Col 1 - navigation bar
                    dbc.Col([sideBar]),
                    # Row 1, col 2 - text description
                    dbc.Col([overview.optionsGraph]),
                    # Row 1, Col 3 - table
                    dbc.Col([overview.graphOverview, overview.spinner_dots]),
                ]
            )
        ]
    )

    # *** MOBILE PAGE ***
    page_mobile_layout = html.Div(
        [
            # Row 1 - body
            dbc.Row([mobile_page])
        ]
    )

    return Layout(
        page_1=page_1_layout,
        page_2=page_2_layout,
        page_3=page_3_layout,
        page_4=page_4_layout,
        page_mobile=page_mobile_layout
    )


def load_page(page, algo):
    return html.Div(
        [
            # layout of the app
            dcc.Location(id="url"),
            html.Div(id="page-content", children=page),
            # Hidden divs to store data
            dcc.Store(id="saved-start-date-page-0", data=algo.min_date),
            dcc.Store(id="saved-end-date-page-0", data=algo.max_date),
            dcc.Store(id="saved-find-fund", data=[]),
            dcc.Store(id="saved-top-performers-names", data=[]),
            dcc.Store(id="saved-top-performers", data="no"),
            dcc.Store(id="saved-combine-top-performers", data="no"),
            dcc.Store(id="saved-top-performers-pct", data=15),
            dcc.Store(id="saved-figure-page-0", data=None),
            dcc.Store(id="saved-start-date-page-1", data=algo.min_date),
            dcc.Store(id="saved-end-date-page-1", data=algo.max_date),
            dcc.Store(id="saved-ml-model", data=""),
            dcc.Store(id="saved-ml-spec", data=""),
            dcc.Store(id="saved-ml-text", data="No selected asset."),
            dcc.Store(id="saved-figure-page-1", data=None),
            dcc.Store(
                id="saved-ai-table",
                data=pd.DataFrame(
                    np.array(
                        [
                            [
                                "No result",
                                "No result",
                                "No result",
                                "No result",
                                "No result",
                            ]
                        ]
                    ),
                    columns=[
                        "Name",
                        "ISIN",
                        "Sharpe Ratio",
                        "Average Annual Returns",
                        "Standard Deviation of Returns",
                    ],
                ).to_dict("records"),
            ),
            dcc.Store(id="saved-split-date", data="2017-07-01"),
            dcc.Store(id="saved-ml-model-back", data=""),
            dcc.Store(id="saved-ml-spec-back", data=2),
            dcc.Store(id="saved-pick-num-back", data=5),
            dcc.Store(id="saved-scen-model-back", data=""),
            dcc.Store(id="saved-scen-spec-back", data=1000),
            dcc.Store(id="saved-benchmark-back", data=[]),
            dcc.Store(id="saved-perf-figure-page-2", data=None),
            dcc.Store(id="saved-comp-figure-page-2", data=None),
            dcc.Store(id="saved-universe-figure-page-2", data=None),
            dcc.Store(id="saved-solver", data=""),
            dcc.Store(id="saved-optimization-model", data=""),
            dcc.Store(id="saved-ml-model-lifecycle", data=""),
            dcc.Store(id="saved-ml-spec-lifecycle", data=2),
            dcc.Store(id="saved-pick-num-lifecycle", data=5),
            dcc.Store(id="saved-scen-model-lifecycle", data=""),
            dcc.Store(id="saved-scen-spec-lifecycle", data=1000),
            dcc.Store(id="saved-glidepaths-figure-page-3", data=None),
            dcc.Store(id="saved-performance-figure-page-3", data=None),
            dcc.Store(id="saved-lifecycle-all-figure-page-3", data=None),
            dcc.Store(id="saved_portfolio_value", data=None),
            dcc.Store(id="saved_yearly_withdraws", data=None),
            dcc.Store(id="saved_risk_preference", data=None),
            dcc.Store(id="saved_end_year", data=2040),
        ]
    )
