import dash_bootstrap_components as dbc
from dash import html

from funnel.dashboard.app_components import (
    graphML,
    graphOverview,
    graphResults,
    mobile_page,
    optionBacktest,
    optionGraph,
    optionML,
    sideBar,
    spinner_backtest,
    spinner_dots,
    spinner_ml,
)

"""
# ----------------------------------------------------------------------------------------------------------------------
# LIFECYCLE
# ----------------------------------------------------------------------------------------------------------------------
"""
page_4_layout = html.Div(
    [
        # Row 1 - body
        dbc.Row(
            [
                # Row 1, Col 1 - navigation bar
                dbc.Col([sideBar]),
                # Row 1, col 2 - text description
                dbc.Col([optionBacktest]),
                # Row 1, Col 3 - table
                dbc.Col([graphResults, spinner_backtest]),
            ]
        )
    ]
)


"""
# ----------------------------------------------------------------------------------------------------------------------
# BACK-TESTING
# ----------------------------------------------------------------------------------------------------------------------
"""
page_3_layout = html.Div(
    [
        # Row 1 - body
        dbc.Row(
            [
                # Row 1, Col 1 - navigation bar
                dbc.Col([sideBar]),
                # Row 1, col 2 - text description
                dbc.Col([optionBacktest]),
                # Row 1, Col 3 - table
                dbc.Col([graphResults, spinner_backtest]),
            ]
        )
    ]
)


"""
# ----------------------------------------------------------------------------------------------------------------------
# AI Feature Selection
# ----------------------------------------------------------------------------------------------------------------------
"""
page_2_layout = html.Div(
    [
        # Row 1 - body
        dbc.Row(
            [
                # Row 1, Col 1 - navigation bar
                dbc.Col([sideBar]),
                # Row 1, col 2 - set-up
                dbc.Col([optionML]),
                # Row 1, Col 3 - table
                dbc.Col([graphML, spinner_ml]),
            ]
        )
    ]
)


"""
# ----------------------------------------------------------------------------------------------------------------------
# MARKET OVERVIEW
# ----------------------------------------------------------------------------------------------------------------------
"""
page_1_layout = html.Div(
    [
        # Row 1 - body
        dbc.Row(
            [
                # Row 1, Col 1 - navigation bar
                dbc.Col([sideBar]),
                # Row 1, col 2 - text description
                dbc.Col([optionGraph]),
                # Row 1, Col 3 - table
                dbc.Col([graphOverview, spinner_dots]),
            ]
        )
    ]
)

"""
# ----------------------------------------------------------------------------------------------------------------------
# MOBILE PAGE
# ----------------------------------------------------------------------------------------------------------------------
"""
page_mobile_layout = html.Div(
    [
        # Row 1 - body
        dbc.Row([mobile_page])
    ]
)
