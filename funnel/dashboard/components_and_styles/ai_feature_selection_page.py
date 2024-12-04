import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from ...models.main import TradeBot
from .styles import (
    DESCRIP_INFO,
    GRAPH_LEFT,
    GRAPH_RIGHT,
    LOADING_STYLE,
    MAIN_TITLE,
    OPTION_BTN,
    OPTION_ELEMENT,
    SUB_TITLE,
)

algo = TradeBot()


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
