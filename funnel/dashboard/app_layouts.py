import dash_bootstrap_components as dbc
from dash import html
from ifunnel.models.main import initialize_bot

from .components_and_styles.ai_feature_selection_page import (
    graphML,
    optionML,
    spinner_ml,
)
from .components_and_styles.backtest_page import (
    graphResults,
    optionBacktest,
    spinner_backtest,
)
from .components_and_styles.general import mobile_page, sideBar
from .components_and_styles.lifecycle_page import (
    options_lifecycle,
    results_lifecycle,
    spinner_lifecycle,
)
from .components_and_styles.market_overview_page import divs as market_overview

algo = initialize_bot()

overview = market_overview(algo)

#
#    (graphOverview,
#    optionGraph,
#    spinner_dots),
#)

# *** LIFECYCLE ***
page_4_layout = html.Div(
    [
        # Row 1 - body
        dbc.Row(
            [
                # Row 1, Col 1 - navigation bar
                dbc.Col([sideBar]),
                # Row 1, col 2 - text description
                dbc.Col([options_lifecycle]),
                # Row 1, Col 3 - table
                dbc.Col([results_lifecycle, spinner_lifecycle]),
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
                dbc.Col([optionBacktest]),
                # Row 1, Col 3 - table
                dbc.Col([graphResults, spinner_backtest]),
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
                dbc.Col([optionML]),
                # Row 1, Col 3 - table
                dbc.Col([graphML, spinner_ml]),
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
