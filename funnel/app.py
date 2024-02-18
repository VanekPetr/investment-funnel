import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import dcc, html

from funnel.dashboard.app_callbacks import get_callbacks
from funnel.dashboard.app_layouts import page_1_layout
from funnel.dashboard.utils import logger
from funnel.models.main import TradeBot

logger.setup_logging()
algo = TradeBot()


def load_page():
    return html.Div(
        [
            # layout of the app
            dcc.Location(id="url"),
            html.Div(id="page-content", children=page_1_layout),
            # Hidden divs to store data
            dcc.Store(id="saved-start-date-page-0", data=algo.min_date),
            dcc.Store(id="saved-end-date-page-0", data=algo.max_date),
            dcc.Store(id="saved-find-fund", data=[]),
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
            dcc.Store(id="saved-lifecycle-figure-page-3", data=None),
            dcc.Store(id="saved_risk_preference", data=[]),
            dcc.Store(id="saved_end_year", data=2040),
        ]
    )


# Initialize the app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

# App layout
app.layout = load_page()
# App callbacks
get_callbacks(app)


if __name__ == "__main__":
    app.run_server(debug=False, dev_tools_hot_reload=False)
