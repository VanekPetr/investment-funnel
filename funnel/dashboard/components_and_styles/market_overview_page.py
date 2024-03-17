import dash_bootstrap_components as dbc
from dash import dcc, html

from funnel.dashboard.components_and_styles.styles import (
    GRAPH_LEFT,
    GRAPH_RIGHT,
    LOADING_STYLE,
    MAIN_TITLE,
    OPTION_BTN,
    OPTION_ELEMENT,
    SUB_TITLE,
)
from funnel.models.main import TradeBot

algo = TradeBot()


optionGraph = html.Div(
    [
        html.H5("Investment Funnel", style=MAIN_TITLE),
        html.P("Selected dates for market overview", style=SUB_TITLE),
        # Date picker for plotting
        dcc.DatePickerRange(
            id="picker-show",
            style=OPTION_ELEMENT,
            min_date_allowed=algo.min_date,
            max_date_allowed=algo.max_date,
        ),
        # Option to search for a fund
        html.P("Find your fund", style=SUB_TITLE),
        dcc.Dropdown(
            id="find-fund",
            options=[{"label": value, "value": value} for value in algo.names],
            placeholder="Select here",
            multi=True,
            style=OPTION_ELEMENT,
        ),
        # Button to plot results
        dbc.Button("Show Plot", id="show", style=OPTION_BTN),
    ],
    style=GRAPH_LEFT,
)

# Table
graphOverview = html.Div(id="dotsFig", style=GRAPH_RIGHT)

spinner_dots = html.Div(
    [
        dcc.Loading(
            id="loading-dots",
            children=[html.Div([html.Div(id="loading-output-dots")])],
            type="circle",
            style=LOADING_STYLE,
            color="black",
        ),
    ]
)
