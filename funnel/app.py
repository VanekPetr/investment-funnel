"""
Main application module for the Investment Funnel dashboard.

This module initializes and configures the Dash application, sets up the layout,
and provides entry points for both production (Gunicorn) and local development.
"""

import dash
import dash_bootstrap_components as dbc
from ifunnel.models.main import initialize_bot

from funnel.nav import get_navbar

# Global object - initialize the algorithm bot for use across the application
algo = initialize_bot()


def create_dash_app():
    """
    Create and configure the Dash application.

    This function initializes a Dash application with the following configurations:
    - Multi-page support enabled
    - Bootstrap styling
    - Responsive viewport
    - Callback exceptions suppressed for dynamic loading

    Returns:
        dash.Dash: Configured Dash application instance
    """
    app = dash.Dash(
        __name__,
        use_pages=True,
        pages_folder="pages",  # <-- adjust as needed
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
        suppress_callback_exceptions=True,
    )

    # Global layout (navbar + dynamic page content)
    app.layout = dbc.Container(
        dbc.Row([
            get_navbar(),  # Left column
            dbc.Col(
                dash.page_container,
                width={"size": 10, "offset": 2},  # Adjust to avoid overlap
                style={"padding": "2rem"}
            )
        ]),
        fluid=True,
        style={"padding": "0"}
    )

    return app


# Gunicorn entry
app = create_dash_app()
server = app.server


# Local dev entry point
def main():  # pragma: no cover
    """
    Run the application for local development.

    This function starts the Dash server with hot-reloading enabled
    for development purposes. It uses port 8222 by default.

    Returns:
        dash.Dash: The running Dash application instance
    """
    app.run(debug=False, dev_tools_hot_reload=True, port=8222)
    return app


if __name__ == "__main__":
    main()  # pragma: no cover
