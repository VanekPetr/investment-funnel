from typing import Any

import flask
from dash.dependencies import Input, Output

from .ai_feature import FeatureInput, FeatureOutput
from .app_layouts import (
    page_1_layout,
    page_2_layout,
    page_3_layout,
    page_4_layout,
    page_mobile_layout,
)
from .backtest import BacktestInputs, BacktestOutputs
from .lifecycle import LifecycleInputs, LifecycleOutputs
from .overview import OverviewInputs, OverviewOutputs


def get_callbacks(app, algo):
    # WHICH WEBPAGE
    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def display_page(pathname: str):
        is_mobile = flask.request.headers.get("User-Agent").lower()
        if "mobile" in is_mobile or "mobi" in is_mobile:
            return page_mobile_layout
        elif pathname == "/":
            return page_1_layout(algo)
        elif pathname == "/page-1":
            return page_2_layout(algo)
        elif pathname == "/page-2":
            return page_3_layout(algo)
        else:
            return page_4_layout(algo)

    # BACK-TESTING
    # -----------------------------------------------------------------------------------------------------------------
    @app.callback(
        LifecycleOutputs.as_output_vector(),
        Input("lifecycle-run", "n_clicks"),
        LifecycleInputs.as_state_vector(),
        prevent_initial_call=True
    )
    def plot_lifecycle(click: int, *args: Any):
        # Map args to LifecycleInputs
        keys = list(LifecycleInputs.model_fields.keys())
        input_values = dict(zip(keys, args))
        inputs = LifecycleInputs(**input_values)

        from .lifecycle import plot_lifecycle
        outputs = plot_lifecycle(algo, inputs)
        return outputs.as_tuple()

    # BACK-TESTING
    # -----------------------------------------------------------------------------------------------------------------
    @app.callback(
        BacktestOutputs.as_output_vector(),
        Input("backtestRun", "n_clicks"),
        BacktestInputs.as_state_vector(),
        prevent_initial_call=True
    )
    def plot_backtest(click: int, *args: Any):
        # Map args to BacktestInputs
        keys = list(BacktestInputs.model_fields.keys())
        input_values = dict(zip(keys, args))
        inputs = BacktestInputs(**input_values)

        from .backtest import plot_backtest
        outputs = plot_backtest(algo, inputs)
        return outputs.as_tuple()

    # AI Feature Selection
    # -----------------------------------------------------------------------------------------------------------------
    # PLOT ML MST GRAPH
    @app.callback(
        FeatureOutput.as_output_vector(),
        [Input("MLRun", "n_clicks")],
        FeatureInput.as_state_vector(),
        prevent_initial_call=True
    )
    def plot_ml(click_ml: int, *args: Any):
        # Map args to FeatureInput
        keys = list(FeatureInput.model_fields.keys())
        input_values = dict(zip(keys, args))
        inputs = FeatureInput(**input_values)

        from .ai_feature import plot_ml
        outputs = plot_ml(algo, inputs)
        return outputs.as_tuple()

    # MARKET OVERVIEW
    # -----------------------------------------------------------------------------------------------------------------
    # PLOT GRAPH WITH DOTS
    @app.callback(
        OverviewOutputs.as_output_vector(),
        [Input("show", "n_clicks")],
        OverviewInputs.as_state_vector(),
        prevent_initial_call=True
    )
    def plot_dots(click: int, *args: Any):
        # Map args to OverviewInputs
        keys = list(OverviewInputs.model_fields.keys())
        input_values = dict(zip(keys, args))
        inputs = OverviewInputs(**input_values)

        from .overview import plot_overview
        outputs = plot_overview(algo, inputs)
        return outputs.as_tuple()
