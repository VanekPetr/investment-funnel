from typing import List, NamedTuple, Optional

import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dcc import Graph
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash.html import Div

from .general import sidebar
from .styles import (
    GRAPH_LEFT,
    GRAPH_RIGHT,
    LOADING_STYLE,
    MAIN_TITLE,
    OPTION_BTN,
    OPTION_ELEMENT,
    SUB_TITLE,
)

MarketOverview = NamedTuple('MarketOverviewPage', [
    ('optionsGraph', html.Div),
    ('graphOverview', html.Div),
    ('spinner_dots', html.Div),
])

def create_market_overview_layout(algo):
    """Create the layout for the market overview page."""
    overview = _divs(algo)

    return html.Div(
        [
            # Row 1 - body
            dbc.Row(
                [
                    # Row 1, Col 1 - navigation bar
                    dbc.Col([sidebar()]),
                    # Row 1, col 2 - text description
                    dbc.Col([overview.optionsGraph]),
                    # Row 1, Col 3 - table
                    dbc.Col([overview.graphOverview, overview.spinner_dots]),
                ]
            )
        ]
    )

def _generate_plot(
    algo,
    start_date: str,
    end_date: str,
    fund_set: Optional[List[str]] = None,
    top_performers: Optional[List[str]] = None,
    top_performers_enabled: bool = False,
    top_performers_pct: float = 15,
):
    """
    Generate a scatter plot visualization of assets with their risk and return characteristics.

    This function creates a scatter plot showing the risk and return characteristics of financial
    products. It can highlight specific funds and top performers based on user inputs.

    Args:
        algo: The algorithm object containing data and methods
        start_date: Start date for analysis
        end_date: End date for analysis
        fund_set: List of funds to highlight in the visualization
        top_performers: List of top performing assets to highlight
        top_performers_enabled: Whether to calculate and highlight top performers
        top_performers_pct: Percentage of top performers to highlight

    Returns:
        Tuple[px.scatter, List[str]]: A plotly scatter plot figure and a list of top performers

    Raises:
        ValueError: If there's an issue with the input parameters or data processing
    """
    fund_set = fund_set if fund_set else []
    top_performers_list = []

    # Validate inputs
    if not start_date or not end_date:
        raise ValueError("Start date and end date must be provided")

    if top_performers_enabled and (top_performers_pct <= 0 or top_performers_pct > 100):
        raise ValueError(f"Top performers percentage must be between 0 and 100, got {top_performers_pct}")

    try:
        # Calculate top performers if enabled
        if top_performers_enabled:
            try:
                top_assets = algo.get_top_performing_assets(
                    time_periods=[(start_date, end_date)],
                    top_percent=top_performers_pct / 100,
                )
                top_performers_list = top_assets
            except Exception as e:
                raise ValueError(f"Failed to calculate top performing assets: {str(e)}")
        elif top_performers:
            top_performers_list = top_performers

        # Generate the plot using algo.plot_dots
        try:
            fig = algo.plot_dots(
                start_date=start_date,
                end_date=end_date,
                fund_set=fund_set,
                top_performers=top_performers_list,
            )
        except Exception as e:
            raise ValueError(f"Failed to generate plot: {str(e)}")

        return fig, top_performers_list
    except Exception as e:
        # Re-raise any exceptions with more context
        raise ValueError(f"Error in _generate_plot: {str(e)}")

def _update_market_overview_plot(
    algo,
    click: int,
    start: str,
    end: str,
    search: Optional[List[str]],
    top_performers: str,
    top_performers_pct: float
) -> Graph | Div:
    """
    Update the market overview visualization based on user inputs.

    This function is called by the update_plot callback in app_callbacks.py when the user
    clicks the "Update Plot" button on the market overview page. It generates a scatter plot
    visualization of assets with their risk and return characteristics.

    Args:
        algo: The algorithm object containing data and methods
        click: Button click event
        start: Start date for analysis
        end: End date for analysis
        search: List of funds to highlight in the visualization
        top_performers: Whether to highlight top performers ("yes" or "no")
        top_performers_pct: Percentage of top performers to highlight

    Returns:
        Tuple containing the generated figure as a Dash component and the list of top performers
    """
    if click:
        try:
            # Validate inputs
            if not start or not end:
                raise ValueError("Start date and end date must be provided")

            if top_performers_pct is not None and (top_performers_pct <= 0 or top_performers_pct > 100):
                raise ValueError(f"Top performers percentage must be between 0 and 100, got {top_performers_pct}")

            # Generate the plot
            fig, top_performers_list = _generate_plot(
                algo,
                start_date=str(start),
                end_date=str(end),
                fund_set=search if search else [],
                top_performers_enabled=top_performers == "yes",
                top_performers_pct=top_performers_pct,
            )

            # Create a graph component from the figure
            generated_figure = dcc.Graph(
                figure=fig,
                style={
                    "position": "absolute",
                    "right": "0%",
                    "bottom": "0%",
                    "top": "0%",
                    "left": "0%",
                },
            )

            return [generated_figure]

        except Exception as e:
            # Create an error message figure
            error_figure = dcc.Graph(
                figure={
                    'data': [],
                    'layout': {
                        'title': 'Error: Unable to generate market overview plot',
                        'annotations': [{
                            'text': f"An error occurred: {str(e)}",
                            'showarrow': False,
                            'font': {'size': 16}
                        }]
                    }
                },
                style={
                    "position": "absolute",
                    "right": "0%",
                    "bottom": "0%",
                    "top": "0%",
                    "left": "0%",
                },
            )

            return [error_figure]

    return [html.Div()]

def _divs(algo) -> MarketOverview:
    """
    Create the UI components for the market overview page.

    This function creates the layout for the market overview page, including
    input controls (date picker, dropdown, radio buttons) and a "Show Plot" button.
    It also generates an initial plot using the default parameters.

    Args:
        algo: The algorithm object containing data and methods

    Returns:
        MarketOverview: A named tuple containing the UI components
    """
    # Default values for the plot
    start_date = algo.min_date
    end_date = algo.max_date
    fund_set = []
    top_performers_enabled = False
    top_performers_pct = 15

    # Generate the initial plot with error handling
    try:
        fig, top_performers_list = _generate_plot(
            algo,
            start_date,
            end_date,
            fund_set,
            None,
            top_performers_enabled,
            top_performers_pct
        )

        # Create a graph component from the figure
        generated_figure = dcc.Graph(
            figure=fig,
            style={
                "position": "absolute",
                "right": "0%",
                "bottom": "0%",
                "top": "0%",
                "left": "0%",
            },
        )
    except Exception as e:
        # Create a simple error message figure if initial plot generation fails
        generated_figure = dcc.Graph(
            figure={
                'data': [],
                'layout': {
                    'title': 'Error: Unable to generate initial market overview plot',
                    'annotations': [{
                        'text': f"An error occurred during initial plot generation: {str(e)}",
                        'showarrow': False,
                        'font': {'size': 16}
                    }]
                }
            },
            style={
                "position": "absolute",
                "right": "0%",
                "bottom": "0%",
                "top": "0%",
                "left": "0%",
            },
        )

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
                start_date=start_date,
                end_date=end_date,
            ),
            # Option to search for a fund
            html.P("Find your fund", style=SUB_TITLE),
            dcc.Dropdown(
                id="find-fund",
                options=[{"label": value, "value": value} for value in algo.names],
                placeholder="Select here",
                multi=True,
                style=OPTION_ELEMENT,
                value=fund_set,
            ),
            html.P("Show top performers for each asset class", style=SUB_TITLE),
            dcc.RadioItems(
                [
                    {"label": "yes", "value": "yes"},
                    {"label": "no", "value": "no"},
                ],
                value="no" if not top_performers_enabled else "yes",
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
                        value=top_performers_pct,
                        style=OPTION_ELEMENT,
                    )
                ],
                style={"display": "none" if not top_performers_enabled else "block"},
            ),
            # Button to update the plot
            dbc.Button("Update Plot", id="show", style=OPTION_BTN),
            # Hidden div to store top performers will be created by callback
        ],
        style=GRAPH_LEFT,
    )

    # Table with the generated figure
    graphOverview = html.Div(id="dotsFig", children=generated_figure, style=GRAPH_RIGHT)

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


    # Return the NamedTuple with the components
    return MarketOverview(
        optionsGraph=optionsGraph,
        graphOverview=graphOverview,
        spinner_dots=spinner_dots,
    )


# You must import this to make the callback do something

def register_callbacks(app, algo):
    @app.callback(
        [
            Output("dotsFig", "children")
        ],
        [Input("show", "n_clicks")],
        [
            State("picker-show", "start_date"),
            State("picker-show", "end_date"),
            State("find-fund", "value"),
            State("top-performers", "value"),
            State("top-performers-pct", "value")
        ],
        prevent_initial_call=True,
    )
    def update_plot(
        click,
        start,
        end,
        search,
        top_performers,
        top_performers_pct
    ):
        if not start or not end:
            raise PreventUpdate

        return _update_market_overview_plot(
            algo,
            click,
            start,
            end,
            search,
            top_performers,
            top_performers_pct
        )

    @app.callback(
        Output(component_id="if-top-performers", component_property="style"),
        Input(component_id="top-performers", component_property="value"),
        prevent_initial_call=True,
    )
    def update_output_top_performers(value):
        """
        Toggle the visibility of the top performers percentage input based on the radio button selection.

        Args:
            value: The value of the top-performers radio button ("yes" or "no")

        Returns:
            dict: A style dictionary with display property set to "block" or "none"
        """
        if value == "yes":
            return {"display": "block"}
        else:
            return {"display": "none"}
