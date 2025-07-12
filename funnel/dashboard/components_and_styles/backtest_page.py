from typing import List, NamedTuple, Optional, Tuple

import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State

from .general import sidebar
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

Backtest = NamedTuple('BacktestPage', [
    ('options', html.Div),
    ('results', html.Div),
    ('spinner', html.Div),
])

def _generate_backtest(
    algo,
    model: str,
    model_spec: int,
    pick_top: int,
    scen_model: str,
    scen_spec: int,
    benchmark: Optional[List[str]],
    start_data: str,
    end_train: str,
    start_test: str,
    end_data: str,
    solver: str,
    optimization_model: str,
    lower_bound: int,
) -> Tuple[dcc.Graph, dcc.Graph, dcc.Graph, List[str]]:
    """
    Generate backtesting analysis figures based on user inputs.

    This function performs ML-based asset selection and backtesting to generate performance,
    composition, and universe comparison figures.

    Args:
        algo: The algorithm object containing data and methods
        model: ML model type (MST or Clustering)
        model_spec: Number of MST runs or clusters
        pick_top: Number of top assets to select from each cluster
        scen_model: Scenario model type
        scen_spec: Number of simulations
        benchmark: Selected benchmark(s)
        start_data: Start date for training data
        end_train: End date for training data
        start_test: Start date for test data
        end_data: End date for test data
        solver: Optimization solver
        optimization_model: Optimization model type
        lower_bound: Minimum required asset weight in portfolio

    Returns:
        Tuple containing performance figure, composition figure, universe figure, and subset of assets
    """
    # Validate model type
    valid_models = ["MST", "Clustering"]
    if model not in valid_models:
        raise ValueError(f"Error in callback plot_backtest: "
                         f"Invalid model type. Expected 'MST' or 'Clustering', got '{model}'.")

    # Validate scenario model type
    valid_scenario_models = ["Bootstrap", "Bootstrapping", "MonteCarlo"]
    if scen_model not in valid_scenario_models:
        raise ValueError(f"Error in callback plot_backtest: "
                         f"It appears that a scenario method other than MonteCarlo, "
                         f"Bootstrap, or Bootstrapping has been chosen. "
                         f"Please check for spelling mistakes. Got '{scen_model}'.")

    # Validate date parameters
    if not start_data or not end_train or not start_test or not end_data:
        raise ValueError("Error in callback plot_backtest: All date parameters must be provided.")

    # Validate benchmark parameter
    if benchmark is None:
        raise ValueError("Error in callback plot_backtest: "
                         "Benchmark parameter cannot be None. "
                         "Please select at least one benchmark.")

    # Validate numeric parameters
    if model_spec <= 0:
        raise ValueError(f"Error in callback plot_backtest: model_spec must be positive, got {model_spec}.")
    if pick_top <= 0:
        raise ValueError(f"Error in callback plot_backtest: pick_top must be positive, got {pick_top}.")
    if scen_spec <= 0:
        raise ValueError(f"Error in callback plot_backtest: scen_spec must be positive, got {scen_spec}.")
    if lower_bound < 0:
        raise ValueError(f"Error in callback plot_backtest: lower_bound must be non-negative, got {lower_bound}.")

    # Initialize
    opt_init = ["Optimal", "Optimal Portfolio", "Optimal Portfolio", 3]
    bench_init = ["Benchmark", "Benchmark Portfolio", "Benchmark Portfolio", 3]

    # RUN ML algo
    try:
        if model == "MST":
            _, subset_of_assets = algo.mst(
                start_date=start_data, end_date=end_train, n_mst_runs=model_spec
            )
        else:
            _, subset_of_assets = algo.clustering(
                start_date=start_data,
                end_date=end_train,
                n_clusters=model_spec,
                n_assets=pick_top,
            )
    except Exception as e:
        raise ValueError(f"Error in callback plot_backtest: Failed to run ML algorithm: {str(e)}")

    # Validate subset_of_assets
    if not subset_of_assets or len(subset_of_assets) == 0:
        raise ValueError("Error in callback plot_backtest: No assets were selected by the ML algorithm.")

    # RUN THE BACKTEST
    # Note: The backtest function requires the specified solver to be installed.
    # If the solver is not installed, it will raise an exception that is caught below
    # and a user-friendly error message will be displayed.
    try:
        opt_table, bench_table, fig_performance, fig_composition = algo.backtest(
            start_train_date=start_data,
            start_test_date=start_test,
            end_test_date=end_data,
            subset_of_assets=subset_of_assets,
            benchmarks=benchmark,
            scenarios_type=scen_model,
            n_simulations=scen_spec,
            model=optimization_model,
            solver=solver,
            lower_bound=lower_bound,
        )
    except TypeError as e:
        error_msg = str(e)
        if "datetime64[ns, UTC]" in error_msg and "NoneType" in error_msg:
            raise ValueError("Error in callback plot_backtest: Invalid date type comparison.")
        else:
            raise ValueError(f"Error in callback plot_backtest: Failed to run backtest: {error_msg}")
    except Exception as e:
        error_msg = str(e)
        if "solver" in error_msg.lower() and "not installed" in error_msg.lower():
            # Handle the case when the selected solver is not installed
            solver_name = solver.lower() if solver else "selected solver"
            raise ValueError(f"Error in callback plot_backtest: The solver {solver_name} is not installed. "
                            f"Please install the {solver_name} package or select a different solver.")
        else:
            raise ValueError(f"Error in callback plot_backtest: Failed to run backtest: {error_msg}")

    # Create graph components from the figures
    perf_figure = dcc.Graph(
        figure=fig_performance, style={"margin": "0%", "height": "800px"}
    )
    comp_figure = dcc.Graph(figure=fig_composition, style={"margin": "0%"})

    # Generate universe comparison figure
    try:
        fig_universe = algo.plot_dots(
            start_date=start_test,
            end_date=end_data,
            optimal_portfolio=opt_table.iloc[0].to_list() + opt_init,
            benchmark=bench_table.iloc[0].to_list() + bench_init,
        )
    except Exception as e:
        raise ValueError(f"Error in callback plot_backtest: Failed to generate universe comparison figure: {str(e)}")

    universe_figure = dcc.Graph(
        figure=fig_universe, style={"margin": "0%", "height": "1200px"}
    )

    return perf_figure, comp_figure, universe_figure, subset_of_assets

def create_backtest_layout(algo):
    """Create the layout for the backtest page."""
    backtest = _divs(algo)

    return html.Div(
        [
            # Row 1 - body
            dbc.Row(
                [
                    # Row 1, Col 1 - navigation bar
                    dbc.Col([sidebar()]),
                    # Row 1, col 2 - text description
                    dbc.Col([backtest.options]),
                    # Row 1, Col 3 - table
                    dbc.Col([backtest.results, backtest.spinner]),
                ]
            )
        ]
    )

def _divs(algo) -> Backtest:
    """
    Create the UI components for the backtesting page.

    This function creates the layout for the backtesting page, including
    input controls (date pickers, dropdowns, sliders) and a "Run Backtest" button.
    The actual backtesting functionality is handled by the plot_backtest callback
    in app_callbacks.py, which is triggered when the user clicks the "Run Backtest" button.

    Args:
        algo: The algorithm object containing data and methods

    Returns:
        Backtest: A named tuple containing the UI components
    """
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
                "MOSEK or CLARABEL are recommended. Make sure the selected solver is installed.",
                style=DESCRIP_INFO,
            ),
            dcc.Dropdown(
                id="select-solver",
                options=[
                    {"label": value, "value": value}
                    for value in ["clarabel", "mosek", "ECOS", "ECOS_BB", "SCS"]
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

    return Backtest(
        options=optionBacktest,
        results=graphResults,
        spinner=spinner_backtest,
    )


def register_callbacks(app, algo):
    """
    Register callbacks for the backtest page.

    This function registers the callback for generating backtesting analysis figures
    when the user clicks the "Run Backtest" button.

    Args:
        app: The Dash application instance
        algo: The algorithm object containing data and methods
    """

    @app.callback(
        Output(
            component_id="slider-trading-sizes-container", component_property="style"
        ),
        [Input(component_id="select-solver", component_property="value")],
    )
    def show_hide_element(solver):
        if solver == "ECOS_BB" or solver == "CPLEX" or solver == "MOSEK":
            return {"display": "block"}
        else:
            return {"display": "none"}

    @app.callback(
        Output("slider-trading-sizes-output", "children"),
        [Input("slider-trading-sizes", "value")],
    )
    def update_trading_sizes(value):
        return "Minimum required asset weight in the portfolio: {}%".format(value)


    # @app.callback(
    #     Output("slider-output-container-backtest", "children"),
    #     [Input("slider-backtest", "value")],
    # )
    # def update_output_cluster(value):
    #     return "In case of CLUSTERING: # of the best performing assets selected from each cluster: {}".format(
    #         value
    #     )

    # @app.callback(
    #     [
    #         Output("picker-test", "start_date"),
    #         Output("picker-test", "end_date"),
    #         Output("picker-train", "start_date"),
    #         Output("picker-train", "end_date"),
    #     ],
    #     Input("picker-train", "end_date"),
    #     prevent_initial_call=True,
    # )
    # def update_test_date(selected_date):
    #     """
    #     Update the test date range based on the selected training end date.
    #
    #     This callback ensures that the training and test periods are properly synchronized.
    #     When the user changes the end date of the training period, the test period is
    #     automatically adjusted to start from that date.
    #
    #     Args:
    #         selected_date: The selected end date for the training period
    #
    #     Returns:
    #         Updated date ranges for the test and training periods
    #     """
    #     if selected_date:
    #         split_date = selected_date
    #     else:
    #         # Use the saved split date (dynamically calculated in app_layouts.py)
    #         split_date
    #
    #     return split_date, algo.max_date, algo.min_date, split_date

    @app.callback(
        Output("slider-output-container-backtest-ml", "children"),
        [Input("slider-backtest-ml", "value")],
    )
    def update_output_ml_type(value):
        return "# of clusters or # of MST runs: {}".format(value)

    @app.callback(
        [
            Output("backtestPerfFig", "children"),
            Output("backtestCompFig", "children"),
            Output("select-ml", "value"),
            Output("slider-backtest-ml", "value"),
            Output("slider-backtest", "value"),
            Output("select-scenarios", "value"),
            Output("my-slider2", "value"),
            Output("select-benchmark", "value"),
            Output("loading-output-backtest", "children"),
            Output("backtestUniverseFig", "children"),
            Output("select-solver", "value"),
            Output("select-optimization-model", "value"),
        ],
        [Input("backtestRun", "n_clicks")],
        [
            State("select-ml", "value"),
            State("slider-backtest-ml", "value"),
            State("slider-backtest", "value"),
            State("select-scenarios", "value"),
            State("my-slider2", "value"),
            State("select-benchmark", "value"),
            State("picker-train", "start_date"),
            State("picker-train", "end_date"),
            State("picker-test", "start_date"),
            State("picker-test", "end_date"),
            State("select-solver", "value"),
            State("select-optimization-model", "value"),
            State("slider-trading-sizes", "value"),
        ],
        prevent_initial_call=True,
    )
    def plot_backtest(
        click,
        model,
        model_spec,
        pick_top,
        scen_model,
        scen_spec,
        benchmark,
        start_data,
        end_train,
        start_test,
        end_data,
        solver,
        optimization_model,
        lower_bound,
    ):
        """
        Generate backtesting analysis figures based on user inputs.

        This callback runs when the user clicks the "Run" button on the backtesting page.
        It uses the generate_backtest function to generate performance,
        composition, and universe comparison figures.

        Args:
            click: Button click event
            model: ML model type (MST or Clustering)
            model_spec: Number of MST runs or clusters
            pick_top: Number of top assets to select from each cluster
            scen_model: Scenario model type
            scen_spec: Number of simulations
            benchmark: Selected benchmark(s)
            start_data: Start date for training data
            end_train: End date for training data
            start_test: Start date for test data
            end_data: End date for test data
            solver: Optimization solver
            optimization_model: Optimization model type
            lower_bound: Minimum required asset weight in portfolio

        Returns:
            Multiple outputs including figures and saved state values
        """
        if not click:
            return (

                True,
            )

        # Generate the plots using the generate_backtest function
        perf_figure, comp_figure, generated_figure, subset_of_assets = _generate_backtest(
            algo,
            model=model,
            model_spec=model_spec,
            pick_top=pick_top,
            scen_model=scen_model,
            scen_spec=scen_spec,
            benchmark=benchmark,
            start_data=start_data,
            end_train=end_train,
            start_test=start_test,
            end_data=end_data,
            solver=solver,
            optimization_model=optimization_model,
            lower_bound=lower_bound,
        )

        return (
            perf_figure,
            comp_figure,
            model,
            model_spec,
            pick_top,
            scen_model,
            scen_spec,
            benchmark,
            model,
            model_spec,
            pick_top,
            scen_model,
            scen_spec,
            benchmark,
            perf_figure,
            comp_figure,
            True,
            generated_figure,
            solver,
            optimization_model
        )
