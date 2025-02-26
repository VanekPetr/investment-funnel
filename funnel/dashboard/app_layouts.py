from typing import NamedTuple

import dash_bootstrap_components as dbc
from dash import html

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
