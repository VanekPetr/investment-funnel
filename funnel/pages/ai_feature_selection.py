from typing import Any

import dash
import dash_bootstrap_components as dbc
from dash import Input, callback, dash_table, dcc, html
from ifunnel.models.main import initialize_bot

from funnel.models.ai_feature import FeatureInput, FeatureOutput, plot_ml
from funnel.styles import (
    DESCRIP_INFO,
    GRAPH_LEFT,
    GRAPH_RIGHT,
    LOADING_STYLE,
    MAIN_TITLE,
    OPTION_BTN,
    OPTION_ELEMENT,
    SUB_TITLE,
)

dash.register_page(__name__, path="/ai_feature_selection", name="AI Feature Selection")

algo = initialize_bot()

# Create the UI components for the AI Feature Selection page
optionML = html.Div(
    [
        html.H5("Minimum Spanning Tree & Clustering", style=MAIN_TITLE),
        html.P(
            "Use machine learning algorithms to decrease the number of ETFs in your asset universe.",
            style=DESCRIP_INFO,
        ),
        html.P("Time period for feature selection", style=SUB_TITLE),
        # Select time period
        dcc.DatePickerRange(
            id="picker-AI",
            style=OPTION_ELEMENT,
            min_date_allowed=algo.min_date,
            max_date_allowed=algo.max_date,
            start_date=algo.min_date,
            end_date=algo.max_date,
        ),
        html.P("AI/ML model", style=SUB_TITLE),
        # Select MST
        dcc.Dropdown(
            id="model-dropdown",
            options=[
                {"label": "Minimum Spanning Tree", "value": "MST"},
                {"label": "Clustering", "value": "Cluster"},
            ],
            placeholder="Select algorithm",
            style=OPTION_ELEMENT,
        ),
        html.P("# of Clusters or # of MST runs", style=SUB_TITLE),
        # Select clustering
        dcc.Dropdown(
            id="ML-num-dropdown",
            options=[
                {"label": "2", "value": 2},
                {"label": "3", "value": 3},
                {"label": "4", "value": 4},
                {"label": "5", "value": 5},
            ],
            placeholder="Select number",
            style=OPTION_ELEMENT,
        ),
        # RUN Clustering
        dbc.Button("Compute", id="MLRun", style=OPTION_BTN),
    ],
    style=GRAPH_LEFT,
)

selectionBar = html.Div(
    [
        html.H5("Selected assets", style={"text-align": "left", "margin-left": "2%"}),
        html.Div(
            id="AInumber",
            style={"text-align": "left", "margin-left": "2%"},
            children="No selected asset.",
        ),
        dash_table.DataTable(
            id="AIResult",
            columns=[
                {"name": "Name", "id": "Name"},
                {"name": "ISIN", "id": "ISIN"},
                {"name": "Sharpe Ratio", "id": "Sharpe Ratio"},
                {"name": "Annual Returns", "id": "Average Annual Returns"},
                {"name": "STD", "id": "Standard Deviation of Returns"},
            ],
            style_table={"width": "48%", "margin": "2%"},
            style_cell={"textAlign": "center"},
            style_as_list_view=True,
            style_header={"fontWeight": "bold"},
            style_cell_conditional=[
                {"if": {"column_id": c}, "textAlign": "left"}
                for c in ["variable", "Group name", "subgroup name", "Attribute text"]
            ],
        ),
    ],
    style=OPTION_ELEMENT,
)

# AI Feature selection graph
graphML = html.Div(
    [html.Div(id="mlFig", style=OPTION_ELEMENT), selectionBar], style=GRAPH_RIGHT
)

spinner_ml = html.Div(
    [
        dcc.Loading(
            id="loading-ml",
            children=[html.Div([html.Div(id="loading-output-ml")])],
            type="circle",
            style=LOADING_STYLE,
            color="black",
        ),
    ]
)

layout = html.Div(
    dbc.Row(
        [
            dbc.Col(optionML, width=4, style={"padding": "2rem"}),
            dbc.Col([graphML, spinner_ml], width=8, style={"padding": "2rem"}),
        ],
        style={"height": "100vh", "overflowY": "auto"},
    )
)

@callback(
    FeatureOutput.as_output_vector(),
    [Input("MLRun", "n_clicks")],
    FeatureInput.as_state_vector(),
    prevent_initial_call=True
)
def run_ml(click: int, *args: Any):
    if not click:
        raise dash.exceptions.PreventUpdate

    keys = list(FeatureInput.model_fields.keys())
    input_values = dict(zip(keys, args))
    inputs = FeatureInput(**input_values)

    outputs = plot_ml(algo, inputs)
    return outputs.as_tuple()
