from typing import List, NamedTuple, Tuple

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

LifeCycle = NamedTuple('LifeCyclePage', [
    ('options', html.Div),
    ('results', html.Div),
    ('spinner', html.Div),
])

def _generate_lifecycle_plot(
    algo,
    model: str,
    model_spec: int,
    pick_top: int,
    scen_model: str,
    scen_spec: int,
    start_data: str,
    end_train: str,
    end_year: int,
    portfolio_value: float,
    yearly_withdraws: float,
    risk_preference: float,
) -> Tuple[dcc.Graph, dcc.Graph, dcc.Graph, List[str]]:
    """
    Generate lifecycle investment analysis figures based on user inputs.

    This function performs ML-based asset selection and lifecycle scenario analysis to generate
    glidepath, performance, and asset allocation figures.

    Args:
        algo: The algorithm object containing data and methods
        model: ML model type (MST or Clustering)
        model_spec: Number of MST runs or clusters
        pick_top: Number of top assets to select from each cluster
        scen_model: Scenario model type
        scen_spec: Number of simulations
        start_data: Start date for analysis
        end_train: End date for training data
        end_year: Final year for lifecycle analysis
        portfolio_value: Initial portfolio value
        yearly_withdraws: Annual withdrawal amount
        risk_preference: Initial risk appetite (percentage)

    Returns:
        Tuple containing glidepaths figure, performance figure, lifecycle all figure, and subset of assets
    """
    # Validate model type
    valid_models = ["MST", "Clustering"]
    if model not in valid_models:
        raise ValueError(f"Error in callback plot_lifecycle: "
                         f"Invalid model type. Expected 'MST' "
                         f"or 'Clustering', got '{model}'.")

    # Validate scenario model type
    valid_scenario_models = ["Bootstrap", "MonteCarlo"]
    if scen_model not in valid_scenario_models:
        raise ValueError(f"Error in callback plot_lifecycle: "
                         f"It appears that a scenario method other than "
                         f"MonteCarlo or Bootstrap has been chosen. "
                         f"Please check for spelling mistakes. Got '{scen_model}'.")

    # Validate numeric parameters
    if model_spec <= 0:
        raise ValueError(f"Error in callback plot_lifecycle: model_spec must be positive, got {model_spec}.")
    if pick_top <= 0:
        raise ValueError(f"Error in callback plot_lifecycle: pick_top must be positive, got {pick_top}.")
    if scen_spec <= 0:
        raise ValueError(f"Error in callback plot_lifecycle: "
                         f"scen_spec must be positive, got {scen_spec}.")
    if portfolio_value <= 0:
        raise ValueError(f"Error in callback plot_lifecycle: "
                         f"portfolio_value must be positive, got {portfolio_value}.")
    if yearly_withdraws < 0:
        raise ValueError(f"Error in callback plot_lifecycle: "
                         f"yearly_withdraws must be non-negative, got {yearly_withdraws}.")
    if risk_preference <= 0 or risk_preference > 100:
        raise ValueError(f"Error in callback plot_lifecycle: "
                         f"risk_preference must be between 0 and 100, got {risk_preference}.")

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
        raise ValueError(f"Error in callback plot_lifecycle: Failed to run ML algorithm: {str(e)}")

    # Validate subset_of_assets
    if not subset_of_assets or len(subset_of_assets) == 0:
        raise ValueError("Error in callback plot_lifecycle: No assets were selected by the ML algorithm.")

    # RUN THE LIFECYCLE FUNCTION
    try:
        _, _, fig_performance, fig_glidepaths, _, _, fig_composition_all = (
            algo.lifecycle_scenario_analysis(
                subset_of_assets=subset_of_assets,
                scenarios_type=scen_model,
                n_simulations=scen_spec,
                end_year=end_year,
                withdrawals=yearly_withdraws,
                initial_risk_appetite=risk_preference / 100,
                initial_budget=portfolio_value,
            )
        )
    except Exception as e:
        # Check if the error is related to gaussian_kde dimensionality issue
        if "singular data covariance matrix" in str(e) and "gaussian_kde" in str(e):
            # Create simple placeholder figures with error message
            fig_performance = dcc.Graph(
                figure={
                    'data': [],
                    'layout': {
                        'title': 'Error: Unable to generate performance plot',
                        'annotations': [{
                            'text': 'The data has insufficient variation for density estimation. '
                                    'Try different parameters or more scenarios.',
                            'showarrow': False,
                            'font': {'size': 16}
                        }]
                    }
                },
                style={"margin": "0%", "height": "800px"}
            )

            fig_glidepaths = dcc.Graph(
                figure={
                    'data': [],
                    'layout': {
                        'title': 'Glidepaths',
                        'annotations': [{
                            'text': 'Glidepath data is available but performance plots could not be generated.',
                            'showarrow': False,
                            'font': {'size': 16}
                        }]
                    }
                },
                style={"margin": "0%", "height": "800px"}
            )

            fig_composition_all = dcc.Graph(
                figure={
                    'data': [],
                    'layout': {
                        'title': 'Error: Unable to generate composition plot',
                        'annotations': [{
                            'text': 'The data has insufficient variation for density estimation. '
                                    'Try different parameters or more scenarios.',
                            'showarrow': False,
                            'font': {'size': 16}
                        }]
                    }
                },
                style={"margin": "0%", "height": "1300px"}
            )

            return fig_glidepaths, fig_performance, fig_composition_all, subset_of_assets
        else:
            # For other errors, raise the original exception
            raise ValueError(f"Error in callback plot_lifecycle: Failed to run lifecycle scenario analysis: {str(e)}")

    performance_figure = dcc.Graph(
        figure=fig_performance, style={"margin": "0%", "height": "800px"}
    )
    glidepaths_figure = dcc.Graph(
        figure=fig_glidepaths, style={"margin": "0%", "height": "800px"}
    )
    lifecycle_all_figure = dcc.Graph(
        figure=fig_composition_all, style={"margin": "0%", "height": "1300px"}
    )

    return glidepaths_figure, performance_figure, lifecycle_all_figure, subset_of_assets


def create_lifecycle_layout(algo):
    """Create the layout for the lifecycle page."""
    lifecycle = _divs(algo)

    return html.Div(
        [
            # Row 1 - body
            dbc.Row(
                [
                    # Row 1, Col 1 - navigation bar
                    dbc.Col([sidebar()]),
                    # Row 1, col 2 - text description
                    dbc.Col([lifecycle.options]),
                    # Row 1, Col 3 - table
                    dbc.Col([lifecycle.results, lifecycle.spinner]),
                ]
            )
        ]
    )

# call this function from outside
def _divs(algo) -> LifeCycle:
    """
    Create the UI components for the lifecycle investments page.

    This function creates the layout for the lifecycle investments page, including
    input controls (date picker, dropdowns, sliders, input fields) and a "Run Simulations" button.
    The actual lifecycle investment analysis functionality is handled by the plot_lifecycle callback
    in app_callbacks.py, which is triggered when the user clicks the "Run Simulations" button.

    Args:
        algo: The algorithm object containing data and methods

    Returns:
        LifeCycle: A named tuple containing the UI components
    """

    options_lifecycle = html.Div(
        [
            # part1
            html.H5("Lifecycle Investments", style=MAIN_TITLE),
            html.P(
                "Test your lifecycle strategy with a selected risk appetite for a given investment period.",
                style=DESCRIP_INFO,
            ),
            html.P("Data for Simulations", style=SUB_TITLE),
            dcc.DatePickerRange(
                id="picker-lifecycle",
                min_date_allowed=algo.min_date,
                max_date_allowed=algo.max_date,
                start_date=algo.min_date,
                end_date=algo.max_date,
                style=OPTION_ELEMENT,
            ),
            html.P("Final year for lifecycle investments", style=SUB_TITLE),
            dcc.Slider(
                2030,
                2070,
                id="slider-final-year-lifecycle",
                step=None,
                marks={
                    2030: "2030",
                    2035: "2035",
                    2040: "2040",
                    2045: "2045",
                    2050: "2050",
                    2055: "2055",
                    2060: "2060",
                    2065: "2065",
                    2070: "2070",
                },
                value=2040,
            ),
            html.P("Feature Selection", style=SUB_TITLE),
            dcc.Dropdown(
                id="select-ml-lifecycle",
                options=[
                    {"label": "Minimum Spanning Tree", "value": "MST"},
                    {"label": "Clustering", "value": "Clustering"},
                ],
                placeholder="Select ML method",
                style=OPTION_ELEMENT,
            ),
            html.P(
                "Number of Clusters or MST runs",
                style=DESCRIP_INFO,
            ),
            html.Div(id="slider-output-container-lifecycle-ml", style=DESCRIP_INFO),
            # part2
            dcc.Slider(id="slider-lifecycle-ml", min=1, max=5, step=1, value=2),
            html.Div(
                id="slider-output-container-lifecycle",
                children=[
                    html.P(
                        "Number of the best performing assets selected from each cluster",
                        style=DESCRIP_INFO,
                    ),
                    dcc.Slider(
                        id="slider-lifecycle",
                        min=1,
                        max=20,
                        step=1,
                        value=2,
                    ),
                ],
                style={"display": "none"},
            ),
            html.P("Scenarios", style=SUB_TITLE),
            dcc.Dropdown(
                id="select-scenarios-lifecycle",
                options=[
                    {"label": "Bootstrapping", "value": "Bootstrap"},
                    {"label": "Monte Carlo", "value": "MonteCarlo"},
                ],
                placeholder="Select scenario generation method",
                style=OPTION_ELEMENT,
            ),
            html.Div(id="slider-output-container-2-lifecycle", style=DESCRIP_INFO),
            # part3
            dcc.Slider(
                250,
                2000,
                id="my-slider-2-lifecycle",
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
            html.P("Initial portfolio value", style=SUB_TITLE),
            dcc.Input(
                id="initial-portfolio-value-lifecycle",
                type="number",
                value=100000,
                style=OPTION_ELEMENT,
            ),
            html.P("Yearly withdraws", style=SUB_TITLE),
            dcc.Input(
                id="yearly-withdraws-lifecycle",
                type="number",
                value=1000,
                style=OPTION_ELEMENT,
            ),
            html.P(
                "Initial risk appetite in terms of standard deviation in %", style=SUB_TITLE
            ),
            dcc.Input(
                id="initial-risk-appetite-lifecycle",
                type="number",
                value=15,
                style=OPTION_ELEMENT,
            ),
            dbc.Button(
                "Run Simulations",
                id="lifecycle-run",
                loading_state={"is_loading": "true"},
                style=OPTION_BTN,
            ),
        ],
        style=GRAPH_LEFT,
    )

    results_lifecycle = html.Div(
        [
            html.Div(id="glidepaths-output-fig", style=OPTION_ELEMENT),
            html.Div(id="performance-output-fig", style=OPTION_ELEMENT),
            html.Div(id="lifecycle-all-output-fig", style=OPTION_ELEMENT),
        ],
        style=GRAPH_RIGHT,
    )

    spinner_lifecycle = html.Div(
        [
            dcc.Loading(
                id="loading-output-lifecycle",
                children=[html.Div([html.Div(id="loading-output-backtest-lifecycle")])],
                type="circle",
                style=LOADING_STYLE,
                color="black",
            ),
        ]
    )

    return LifeCycle(
        options=options_lifecycle,
        results=results_lifecycle,
        spinner=spinner_lifecycle
    )


def register_callbacks(app, algo):
    """
    Register callbacks for the lifecycle page.

    This function registers the callback for generating lifecycle investment analysis figures
    when the user clicks the "Run Simulations" button.

    Args:
        app: The Dash application instance
        algo: The algorithm object containing data and methods
    """
    # @app.callback(
    #     Output("slider-output-container-2-lifecycle", "children"),
    #     Input("my-slider-2-lifecycle", "value"),
    # )
    # def update_output_lifecycle(value):
    #     return "# of scenarios: {}".format(value)

    @app.callback(
        Output(
            component_id="slider-output-container-lifecycle", component_property="style", allow_duplicate=True
        ),
        Input(component_id="select-ml-lifecycle", component_property="value"),
        prevent_initial_call=True
    )
    def update_output_cluster_lifecycle(value):
        if value == "Clustering":
            return {"display": "block"}
        else:
            return {"display": "none"}

    @app.callback(
        Output("slider-output-container-lifecycle-ml", "children"),
        [Input("slider-lifecycle-ml", "value")],
    )
    def update_output_ml_type_lifecycle(value):
        return "# of clusters or # of MST runs: {}".format(value)


    @app.callback(
        [
            Output("glidepaths-output-fig", "children"),
            Output("performance-output-fig", "children"),
            Output("lifecycle-all-output-fig", "children"),
            Output("loading-output-lifecycle", "children"),
        ],
        [Input("lifecycle-run", "n_clicks")],
        [
            State("select-ml-lifecycle", "value"),
            State("slider-lifecycle-ml", "value"),
            State("slider-lifecycle", "value"),
            State("select-scenarios-lifecycle", "value"),
            State("my-slider-2-lifecycle", "value"),
            State("picker-lifecycle", "start_date"),
            State("picker-lifecycle", "end_date"),
            State("slider-final-year-lifecycle", "value"),
            State("initial-portfolio-value-lifecycle", "value"),
            State("yearly-withdraws-lifecycle", "value"),
            State("initial-risk-appetite-lifecycle", "value"),
        ],
        prevent_initial_call=True
    )
    def plot_lifecycle(
        click,
        model,
        model_spec,
        pick_top,
        scen_model,
        scen_spec,
        start_data,
        end_train,
        end_year,
        portfolio_value,
        yearly_withdraws,
        risk_preference
    ):
        """
        Generate lifecycle investment analysis figures based on user inputs.

        This callback runs when the user clicks the "Run" button on the lifecycle page.
        It uses the generate_lifecycle_plot function to generate glidepath, performance,
        and asset allocation figures.

        Args:
            click: Button click event
            model: ML model type (MST or Clustering)
            model_spec: Number of MST runs or clusters
            pick_top: Number of top assets to select from each cluster
            scen_model: Scenario model type
            scen_spec: Number of simulations
            start_data: Start date for analysis
            end_train: End date for training data
            end_year: Final year for lifecycle analysis
            portfolio_value: Initial portfolio value
            yearly_withdraws: Annual withdrawal amount
            risk_preference: Initial risk appetite (percentage)

        Returns:
            Multiple outputs including figures and a value for the loading output
        """
        # Lifecycle analysis
        if click:
            # Generate the plots using the generate_lifecycle_plot function
            glidepaths_figure, performance_figure, lifecycle_all_figure, _ = _generate_lifecycle_plot(
                algo,
                model=model,
                model_spec=model_spec,
                pick_top=pick_top,
                scen_model=scen_model,
                scen_spec=scen_spec,
                start_data=start_data,
                end_train=end_train,
                end_year=end_year,
                portfolio_value=portfolio_value,
                yearly_withdraws=yearly_withdraws,
                risk_preference=risk_preference,
            )

            # Return the generated figures and a value for the loading output
            return glidepaths_figure, performance_figure, lifecycle_all_figure, True

        # If no click event, return empty figures
        return html.Div(), html.Div(), html.Div(), True
