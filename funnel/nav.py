"""
Navigation module for the Investment Funnel dashboard.

This module provides the navigation sidebar component used across the application.
"""

import dash_bootstrap_components as dbc
from dash import html


def get_navbar():
    """
    Create and return the navigation sidebar component.

    This function builds a responsive sidebar with the application logo and
    navigation links to all available pages. The sidebar is fixed to the left
    side of the screen and styled with Bootstrap.

    Returns:
        dbc.Col: A Bootstrap column component containing the navigation sidebar
    """
    return dbc.Col(
        [
            # Logo (replace with actual image if you have one)
            html.Div(
                html.Img(
                    src="/assets/logo.png",  # Make sure you have this file in `assets/`
                    style={"width": "100%", "padding": "1rem"}
                ),
                style={"textAlign": "center"}
            ),
            # Navigation links
            html.Div(
                [
                    dbc.Nav(
                        [
                            dbc.NavLink("Overview", href="/overview", active="exact"),
                            dbc.NavLink("Lifecycle", href="/lifecycle", active="exact"),
                            dbc.NavLink("Backtest", href="/backtest", active="exact"),
                            dbc.NavLink("AI Feature Selection", href="/ai_feature_selection", active="exact"),
                        ],
                        vertical=True,
                        pills=True,
                    )
                ],
                style={"padding": "1rem"}
            )
        ],
        width=2,
        style={
            "backgroundColor": "#212529",
            "height": "100vh",
            "position": "fixed",
            "left": 0,
            "top": 0,
            "bottom": 0,
            "paddingTop": "1rem",
            "borderRight": "1px solid #343a40"
        }
    )
