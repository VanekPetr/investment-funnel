import dash
import dash_bootstrap_components as dbc
from ifunnel.models.main import initialize_bot

from .dashboard.app_callbacks import get_callbacks
from .dashboard.app_layouts import divs as layout_divs
from .dashboard.app_layouts import load_page


# Function to create the Dash app
def create_app(algo=None):
    # Initialize the bot algorithm if not provided
    algo = algo or initialize_bot()

    # Initialize the Dash app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
        ],
        suppress_callback_exceptions=True,
    )

    # Load the layout
    layout = layout_divs(algo)

    # Set up callbacks
    get_callbacks(app, layout, algo)

    # Set the app layout
    app.layout = load_page(page=layout.page_1)

    # Return the Flask server
    return app


# Create the Dash app instance
app = create_app()

# Create the Flask server instance for Gunicorn
server = app.server

def main():  # pragma: no cover
    app.run_server(debug=True)

# Development server setup
if __name__ == "__main__":     # pragma: no cover
    main()
