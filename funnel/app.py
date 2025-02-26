import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import dcc, html
from ifunnel.models.main import initialize_bot

from funnel.dashboard.app_callbacks import get_callbacks
from funnel.dashboard.app_layouts import divs as layout_divs
from funnel.dashboard.utils import logger

logger.setup_logging()


def load_page(page, algo):
    return html.Div(
        [
            # layout of the app
            dcc.Location(id="url"),
            html.Div(id="page-content", children=page),
            # Hidden divs to store data
            dcc.Store(id="saved-start-date-page-0", data=algo.min_date),
            dcc.Store(id="saved-end-date-page-0", data=algo.max_date),
            dcc.Store(id="saved-find-fund", data=[]),
            dcc.Store(id="saved-top-performers-names", data=[]),
            dcc.Store(id="saved-top-performers", data="no"),
            dcc.Store(id="saved-combine-top-performers", data="no"),
            dcc.Store(id="saved-top-performers-pct", data=15),
            dcc.Store(id="saved-figure-page-0", data=None),
            dcc.Store(id="saved-start-date-page-1", data=algo.min_date),
            dcc.Store(id="saved-end-date-page-1", data=algo.max_date),
            dcc.Store(id="saved-ml-model", data=""),
            dcc.Store(id="saved-ml-spec", data=""),
            dcc.Store(id="saved-ml-text", data="No selected asset."),
            dcc.Store(id="saved-figure-page-1", data=None),
            dcc.Store(
                id="saved-ai-table",
                data=pd.DataFrame(
                    np.array(
                        [
                            [
                                "No result",
                                "No result",
                                "No result",
                                "No result",
                                "No result",
                            ]
                        ]
                    ),
                    columns=[
                        "Name",
                        "ISIN",
                        "Sharpe Ratio",
                        "Average Annual Returns",
                        "Standard Deviation of Returns",
                    ],
                ).to_dict("records"),
            ),
            dcc.Store(id="saved-split-date", data="2017-07-01"),
            dcc.Store(id="saved-ml-model-back", data=""),
            dcc.Store(id="saved-ml-spec-back", data=2),
            dcc.Store(id="saved-pick-num-back", data=5),
            dcc.Store(id="saved-scen-model-back", data=""),
            dcc.Store(id="saved-scen-spec-back", data=1000),
            dcc.Store(id="saved-benchmark-back", data=[]),
            dcc.Store(id="saved-perf-figure-page-2", data=None),
            dcc.Store(id="saved-comp-figure-page-2", data=None),
            dcc.Store(id="saved-universe-figure-page-2", data=None),
            dcc.Store(id="saved-solver", data=""),
            dcc.Store(id="saved-optimization-model", data=""),
            dcc.Store(id="saved-ml-model-lifecycle", data=""),
            dcc.Store(id="saved-ml-spec-lifecycle", data=2),
            dcc.Store(id="saved-pick-num-lifecycle", data=5),
            dcc.Store(id="saved-scen-model-lifecycle", data=""),
            dcc.Store(id="saved-scen-spec-lifecycle", data=1000),
            dcc.Store(id="saved-glidepaths-figure-page-3", data=None),
            dcc.Store(id="saved-performance-figure-page-3", data=None),
            dcc.Store(id="saved-lifecycle-all-figure-page-3", data=None),
            dcc.Store(id="saved_portfolio_value", data=None),
            dcc.Store(id="saved_yearly_withdraws", data=None),
            dcc.Store(id="saved_risk_preference", data=None),
            dcc.Store(id="saved_end_year", data=2040),
        ]
    )


# Function to create the Dash app
def create_app(algo=None):
    # Initialize the app
    algo = algo or initialize_bot()

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"}
        ],
    )

    layout = layout_divs(algo)

    # Set up callbacks
    get_callbacks(app, layout, algo)

    # Set the app layout
    app.layout = load_page(page=layout.page_1, algo=algo)

    # Return the Flask server
    return app.server

# Create the Flask server instance for Gunicorn
server = create_app()

# Development server setup
if __name__ == "__main__":
    app = dash.Dash(__name__)
    # Initialize the app
    algo = initialize_bot()

    app.server = create_app(algo=algo)
    app.run_server(debug=True, dev_tools_hot_reload=False)
