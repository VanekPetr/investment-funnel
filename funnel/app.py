import dash
import dash_bootstrap_components as dbc
from ifunnel.models.main import initialize_bot

from .dashboard.nav import get_navbar

# Global object
algo = initialize_bot()


def create_dash_app():
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


# Local dev
def main():
    app.run(debug=False, dev_tools_hot_reload=True, port=8222)
    return app


if __name__ == "__main__":
    main()  # pragma: no cover
