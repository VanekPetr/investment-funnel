import base64
import os
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import html

from .styles import (
    SIDEBAR_STYLE,
)

ROOT_DIR = Path(__file__).parent.parent.parent
encoded_image = base64.b64encode(
    open(os.path.join(ROOT_DIR, "assets/ALGO_logo.png"), "rb").read()
)

# sidebar with navigation
sideBar = html.Div(
    [
        html.Img(
            src="data:image/png;base64,{}".format(encoded_image.decode()),
            style={"position": "fixed", "width": "9%", "margin-top": "16px"},
        ),
        html.H5(
            "Investment Funnel",
            style={"color": "#ffd0b3", "position": "fixed", "top": "7%"},
        ),
        dbc.Nav(
            [
                dbc.NavLink("Market Overview", id="page0", href="/", active="exact"),
                dbc.NavLink(
                    "AI Feature Selection",
                    id="page1",
                    href="/page-1",
                    active="exact",
                    n_clicks=0,
                ),
                dbc.NavLink("Backtesting", id="page2", href="/page-2", active="exact"),
                dbc.NavLink(
                    "Lifecycle Investments", id="page3", href="/page-3", active="exact"
                ),
            ],
            vertical=True,
            pills=True,
            style={"position": "fixed", "top": "9%"},
        ),
    ],
    style=SIDEBAR_STYLE,
)
