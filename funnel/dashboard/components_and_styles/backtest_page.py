import cvxpy
import dash_bootstrap_components as dbc
from dash import dcc, html

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


optionBacktest = html.Div(
    [
        # part1
        html.H5("Backtesting", style=MAIN_TITLE),
        html.P(
            "Test your investment strategy with a selected ML model, scenario generation method and the CVaR model "
            + "for a given training and testing time period",
            style=DESCRIP_INFO,
        ),
        html.P("Training period", style=SUB_TITLE),
        dcc.DatePickerRange(
            id="picker-train",
            min_date_allowed=algo.min_date,
            max_date_allowed=algo.max_date,
            start_date=algo.min_date,
            style=OPTION_ELEMENT,
        ),
        html.P("Testing period for backtest", style=SUB_TITLE),
        dcc.DatePickerRange(
            id="picker-test",
            min_date_allowed=algo.min_date,
            max_date_allowed=algo.max_date,
            end_date=algo.max_date,
            style=OPTION_ELEMENT,
        ),
        html.P("Portfolio Optimization Model", style=SUB_TITLE),
        dcc.Dropdown(
            id="select-optimization-model",
            options=[
                {"label": value, "value": value}
                for value in ["CVaR model", "Markowitz model"]
            ],
            placeholder="Select portfolio optimization model",
            style=OPTION_ELEMENT,
        ),
        html.P("Solver", style=SUB_TITLE),
        html.P(
            "MOSEK or CLARABEL are recommended",
            style=DESCRIP_INFO,
        ),
        dcc.Dropdown(
            id="select-solver",
            options=[
                {"label": value, "value": value}
                for value in cvxpy.installed_solvers()
                if value not in ["ECOS", "ECOS_BB", "SCS"]
            ],
            placeholder="Select your installed solver",
            style=OPTION_ELEMENT,
        ),
        html.P("Feature Selection", style=SUB_TITLE),
        dcc.Dropdown(
            id="select-ml",
            options=[
                {"label": "Minimum Spanning Tree", "value": "MST"},
                {"label": "Clustering", "value": "Clustering"},
            ],
            placeholder="Select ML method",
            style=OPTION_ELEMENT,
        ),
        html.Div(id="slider-output-container-backtest-ml", style=DESCRIP_INFO),
        # part2
        dcc.Slider(id="slider-backtest-ml", min=1, max=5, step=1, value=2),
        dcc.Slider(
            id="slider-backtest",
            min=1,
            max=20,
            step=1,
            value=2,
        ),
        html.Div(
            id="slider-trading-sizes-container",
            children=[
                html.Div(id="slider-output-container-backtest", style=DESCRIP_INFO),
                html.P("Trading Sizes", style=SUB_TITLE),
                html.P(
                    "Select the lower bound for the allocated proportion of each selected asset",
                    style=DESCRIP_INFO,
                ),
                html.Div(id="slider-trading-sizes-output", style=DESCRIP_INFO),
                # Create element to hide/show, in this case a slider
                dcc.Slider(id="slider-trading-sizes", min=0, max=10, step=1, value=0),
            ],
            style={"display": "none"},
        ),
        html.P("Scenarios", style=SUB_TITLE),
        dcc.Dropdown(
            id="select-scenarios",
            options=[
                {"label": "Bootstrapping", "value": "Bootstrapping"},
                {"label": "Monte Carlo", "value": "MonteCarlo"},
            ],
            placeholder="Select scenario generation method",
            style=OPTION_ELEMENT,
        ),
        html.Div(id="slider-output-container2", style=DESCRIP_INFO),
        # part3
        dcc.Slider(
            250,
            2000,
            id="my-slider2",
            step=None,
            marks={
                250: "0.25k",
                500: "0.5k",
                750: "0.75k",
                1000: "1k",
                1250: "1.25k",
                1500: "1.5k",
                1750: "1.75k",
                2000: "2k",
            },
            value=1000,
        ),
        html.P("Benchmark", style=SUB_TITLE),
        dcc.Dropdown(
            id="select-benchmark",
            options=[{"label": value, "value": value} for value in algo.names],
            placeholder="Select your ETF benchmark",
            multi=True,
            style=OPTION_ELEMENT,
        ),
        dbc.Button(
            "Run Backtest",
            id="backtestRun",
            loading_state={"is_loading": "true"},
            style=OPTION_BTN,
        ),
    ],
    style=GRAPH_LEFT,
)

# Performance
graphResults = html.Div(
    [
        html.Div(id="backtestPerfFig", style=OPTION_ELEMENT),
        html.Div(id="backtestCompFig", style=OPTION_ELEMENT),
        html.Div(id="backtestUniverseFig", style=OPTION_ELEMENT),
    ],
    style=GRAPH_RIGHT,
)

spinner_backtest = html.Div(
    [
        dcc.Loading(
            id="loading-backtest",
            children=[html.Div([html.Div(id="loading-output-backtest")])],
            type="circle",
            style=LOADING_STYLE,
            color="black",
        ),
    ]
)
