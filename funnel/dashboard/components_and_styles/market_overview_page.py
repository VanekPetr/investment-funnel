import dash_bootstrap_components as dbc
from dash import dcc, html

from ...models.main import TradeBot
from .styles import (
    GRAPH_LEFT,
    GRAPH_RIGHT,
    LOADING_STYLE,
    MAIN_TITLE,
    OPTION_BTN,
    OPTION_ELEMENT,
    SUB_TITLE,
)

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
        html.P("Show top performers for each asset class", style=SUB_TITLE),
        dcc.RadioItems(
            [
                {"label": "yes", "value": "yes"},
                {"label": "no", "value": "no"},
            ],
            inline=True,
            style=OPTION_ELEMENT,
            id="top-performers",
        ),
        html.Div(
            id="if-top-performers",
            children=[
                html.P(
                    "% of top performing assets from each risk class", style=SUB_TITLE
                ),
                dcc.Input(
                    id="top-performers-pct",
                    type="number",
                    value=15,
                    style=OPTION_ELEMENT,
                ),
                html.P("Combine with previous top performers", style=SUB_TITLE),
                dcc.RadioItems(
                    [
                        {"label": "yes", "value": "yes"},
                        {"label": "no", "value": "no"},
                    ],
                    "no",
                    inline=True,
                    style=OPTION_ELEMENT,
                    id="combine-top-performers",
                ),
            ],
            style={"display": "none"},
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
