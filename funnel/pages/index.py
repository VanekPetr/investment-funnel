"""
Root page module for the Investment Funnel dashboard.

This module registers the root URL ("/") and redirects to the Overview page.
"""

import dash
from dash import dcc, html

# Register the root URL and redirect to the Overview page
dash.register_page(
    __name__,
    path="/",
    title="Investment Funnel - Overview"
)

# Define a layout that redirects to the Overview page
layout = html.Div([
    dcc.Location(id="url-redirect", pathname="/overview", refresh=True)
])
