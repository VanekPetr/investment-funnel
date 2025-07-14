"""
Market Overview page module for the Investment Funnel dashboard.

This module defines the UI components and callback functions for the Overview page.
It allows users to visualize the investment universe, search for specific funds,
and highlight top-performing assets across different asset classes.
"""

from typing import Any

import dash
import dash_bootstrap_components as dbc
from dash import Input, callback, dcc, html
from ifunnel.models.main import initialize_bot

from funnel.pages.models.overview import OverviewInputs, OverviewOutputs, plot_overview
from funnel.pages .styles import (
    GRAPH_LEFT,
    GRAPH_RIGHT,
    LOADING_STYLE,
    MAIN_TITLE,
    OPTION_BTN,
    OPTION_ELEMENT,
    SUB_TITLE,
)

dash.register_page(__name__, path="/overview", name="Overview")

algo = initialize_bot()

# Create the UI components for the Market Overview page
optionsGraph = html.Div(
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
            value="no",
            inline=True,
            style=OPTION_ELEMENT,
            id="top-performers",
        ),
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

layout = html.Div(
    dbc.Row(
        [
            dbc.Col(optionsGraph, width=4, style={"padding": "2rem"}),
            dbc.Col([graphOverview, spinner_dots], width=8, style={"padding": "2rem"}),
        ],
        style={"height": "100vh", "overflowY": "auto"},
    )
)

@callback(
    OverviewOutputs.as_output_vector(),
    [Input("show", "n_clicks")],
    OverviewInputs.as_state_vector(),
    prevent_initial_call=True
)
def plot_dots(click: int, *args: Any):
    """
    Callback function for the Market Overview page.

    This function is triggered when the user clicks the "Show Plot" button.
    It processes the input parameters, generates a scatter plot visualization
    of the investment universe, and highlights specific funds and top-performing
    assets based on the user's selections.

    Args:
        click (int): Number of times the button has been clicked
        *args (Any): Variable length argument list containing the input values
                    from the UI components, in the order defined by OverviewInputs.as_state_vector()

    Returns:
        tuple: A tuple of output values for the UI components, as defined by
               OverviewOutputs.as_output_vector()

    Raises:
        dash.exceptions.PreventUpdate: If the callback is triggered without a button click
    """
    if not click:
        raise dash.exceptions.PreventUpdate

    keys = list(OverviewInputs.model_fields.keys())
    input_values = dict(zip(keys, args))
    inputs = OverviewInputs(**input_values)

    outputs = plot_overview(algo, inputs)
    return outputs.as_tuple()
