from typing import List, NamedTuple, Tuple

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html
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

AiFeature = NamedTuple('AiFeaturePage', [
    ('options', html.Div),
    ('selection', html.Div),
    ('graph', html.Div),
    ('spinner', html.Div),
])

def create_ai_feature_selection_layout(algo):
    """Create the layout for the AI feature selection page."""
    ai_feature_selection = _divs(algo)

    return html.Div(
        [
            # Row 1 - body
            dbc.Row(
                [
                    # Row 1, Col 1 - navigation bar
                    dbc.Col([sidebar()]),
                    # Row 1, col 2 - set-up
                    dbc.Col([ai_feature_selection.options]),
                    # Row 1, Col 3 - table
                    dbc.Col([ai_feature_selection.graph, ai_feature_selection.spinner]),
                ]
            )
        ]
    )

def generate_plot_ml(
    algo,
    model: str,
    spec: int,
    start_date: str,
    end_date: str,
) -> Tuple[dcc.Graph, List[str], dict]:
    """
    Generate AI feature selection analysis based on user inputs.

    This function performs either MST (Minimum Spanning Tree) or Clustering analysis on the assets
    and returns the results as a graph and a table of selected assets with their statistics.

    Args:
        algo: The algorithm object containing data and methods
        model: ML model type (MST or Clustering)
        spec: Number of MST runs or clusters
        start_date: Start date for analysis
        end_date: End date for analysis

    Returns:
        Tuple containing the generated figure, subset of assets, and table data
    """
    # Validate model type
    valid_models = ["MST", "Cluster"]
    if model not in valid_models:
        raise ValueError(f"Error in callback plot_ml: Invalid model type. Expected 'MST' or 'Cluster', got '{model}'.")

    # Validate numeric parameters
    if spec <= 0:
        raise ValueError(f"Error in callback plot_ml: spec must be positive, got {spec}.")

    # Validate date parameters
    if not start_date or not end_date:
        raise ValueError("Error in callback plot_ml: All date parameters must be provided.")

    try:
        # MST
        if model == "MST":
            # RUN THE MINIMUM SPANNING TREE METHOD
            fig, ai_subset = algo.mst(
                start_date=start_date,
                end_date=end_date,
                n_mst_runs=spec,
                plot=True,
            )
            generated_figure = dcc.Graph(
                figure=fig, style={"height": "800px", "margin": "0%"}
            )
        # CLUSTERING
        else:
            fig, ai_subset = algo.clustering(
                start_date=start_date,
                end_date=end_date,
                n_clusters=spec,
                n_assets=10,
                plot=True,
            )
            generated_figure = dcc.Graph(
                figure=fig, style={"height": "800px", "margin": "0%"}
            )

        # Get statistics for selected assets
        ai_data = algo.get_stat(start_date=start_date, end_date=end_date)
        ai_table = ai_data.loc[
            list(ai_subset),
            [
                "Name",
                "ISIN",
                "Sharpe Ratio",
                "Average Annual Returns",
                "Standard Deviation of Returns",
            ],
        ]
        # ROUNDING
        ai_table["Standard Deviation of Returns"] = round(
            ai_table["Standard Deviation of Returns"], 2
        )
        ai_table["Average Annual Returns"] = round(
            ai_table["Average Annual Returns"], 2
        )

        return generated_figure, ai_subset, ai_table.to_dict("records")
    except Exception as e:
        # Handle any errors that occur during the analysis
        raise ValueError(f"Error in callback plot_ml: Failed to run AI feature selection: {str(e)}")


def _divs(algo) -> AiFeature:
    """
    Create the UI components for the AI feature selection page.

    This function creates the layout for the AI feature selection page, including
    input controls (date picker, dropdown) and a "Compute" button.
    The actual AI feature selection functionality is handled by the plot_ml callback
    in the register_callbacks function, which is triggered when the user clicks the "Compute" button.

    Args:
        algo: The algorithm object containing data and methods

    Returns:
        AiFeature: A named tuple containing the UI components
    """
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

    return AiFeature(
        options=optionML,
        selection=selectionBar,
        graph=graphML,
        spinner=spinner_ml
    )


def register_callbacks(app, algo):
    """
    Register callbacks for the AI feature selection page.

    This function registers the callback for generating AI feature selection analysis
    when the user clicks the "Compute" button.

    Args:
        app: The Dash application instance
        algo: The algorithm object containing data and methods
    """
    @app.callback(
        [
            Output("mlFig", "children"),
            Output("picker-AI", "start_date"),
            Output("picker-AI", "end_date"),
            Output("AIResult", "data"),
            Output("AInumber", "children"),
            Output("model-dropdown", "value"),
            Output("ML-num-dropdown", "value"),
            Output("saved-start-date-page-1", "data"),
            Output("saved-end-date-page-1", "data"),
            Output("saved-ai-table", "data"),
            Output("saved-ml-model", "data"),
            Output("saved-ml-spec", "data"),
            Output("saved-ml-text", "data"),
            Output("saved-figure-page-1", "data"),
            Output("loading-output-ml", "children"),
        ],
        [Input("url", "pathname"), Input("MLRun", "n_clicks")],
        [
            State("model-dropdown", "value"),
            State("ML-num-dropdown", "value"),
            State("picker-AI", "start_date"),
            State("picker-AI", "end_date"),
            State("saved-start-date-page-1", "data"),
            State("saved-end-date-page-1", "data"),
            State("saved-ai-table", "data"),
            State("saved-ml-model", "data"),
            State("saved-ml-spec", "data"),
            State("saved-ml-text", "data"),
            State("saved-figure-page-1", "data"),
        ],
    )
    def plot_ml(
        pathname,
        click_ml,
        model,
        spec,
        start,
        end,
        saved_start,
        saved_end,
        saved_ai_table,
        saved_model,
        saved_spec,
        saved_text,
        saved_figure,
    ):
        """
        Generate AI feature selection analysis based on user inputs.

        This callback runs when the user clicks the "Run" button on the AI feature selection page.
        It performs either MST (Minimum Spanning Tree) or Clustering analysis on the assets
        and displays the results as a graph and a table of selected assets with their statistics.

        Args:
            pathname: Current URL pathname
            click_ml: Button click event
            model: ML model type (MST or Clustering)
            spec: Number of MST runs or clusters
            start: Start date for analysis
            end: End date for analysis
            saved_start: Previously saved start date
            saved_end: Previously saved end date
            saved_ai_table: Previously saved AI table data
            saved_model: Previously saved model type
            saved_spec: Previously saved model specification
            saved_text: Previously saved text description
            saved_figure: Previously saved figure

        Returns:
            Multiple outputs including the generated figure, date ranges, table data,
            and other state information
        """
        # Only process if we're on the AI feature selection page or if the button wasn't clicked
        if pathname != "/page-1" or not click_ml:
            return (
                saved_figure,
                saved_start,
                saved_end,
                saved_ai_table,
                saved_text,
                saved_model,
                saved_spec,
                saved_start,
                saved_end,
                saved_ai_table,
                saved_model,
                saved_spec,
                saved_text,
                saved_figure,
                True,
            )

        if click_ml:
            selected_start = str(start)
            selected_end = str(end)

            # Generate the plots using the generate_plot_ml function
            generated_figure, ai_subset, ai_table_records = generate_plot_ml(
                algo,
                model=model,
                spec=spec,
                start_date=selected_start,
                end_date=selected_end,
            )

            # Create descriptive text
            ml_text = "Number of selected assets: " + str(len(ai_table_records))

            return (
                generated_figure,
                selected_start,
                selected_end,
                ai_table_records,
                ml_text,
                model,
                spec,
                selected_start,
                selected_end,
                ai_table_records,
                model,
                spec,
                ml_text,
                generated_figure,
                True,
            )
        else:
            return (
                saved_figure,
                saved_start,
                saved_end,
                saved_ai_table,
                saved_text,
                saved_model,
                saved_spec,
                saved_start,
                saved_end,
                saved_ai_table,
                saved_model,
                saved_spec,
                saved_text,
                saved_figure,
                True,
            )
