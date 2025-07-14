import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from ifunnel.models.main import initialize_bot

from .dashboard.app_callbacks import get_callbacks
from .dashboard.app_layouts import page_1_layout

algo = initialize_bot()


def load_page():
    return html.Div(
        [
            # layout of the app
            dcc.Location(id="url"),
            html.Div(id="page-content", children=page_1_layout(algo))
        ]
    )


def create_app():
    # Initialize the app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
        ],
    )
    # App layout
    app.layout = load_page()

    # App callbacks
    get_callbacks(app, algo)

    # return the flask server
    return app.server


# Create the Flask server instance for Gunicorn
server = create_app()


# Keep the development server setup
def main():
    a = create_app()
    dash_app = dash.Dash(__name__)
    dash_app.server = a
    dash_app.run(debug=False, dev_tools_hot_reload=False)
    return

if __name__ == "__main__":
    app = main()            # pragma: no cover
