import base64
import logging
import os
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import html

from .styles import (
    MOBILE_PAGE,
    SIDEBAR_STYLE,
)

# Set up logging
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent.parent
try:
    logo_path = os.path.join(ROOT_DIR, "assets/ALGO_logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            encoded_image = base64.b64encode(f.read())
    else:
        logger.warning(f"Logo file not found at {logo_path}")
        encoded_image = None
except Exception as e:
    logger.error(f"Error loading logo: {e}")
    encoded_image = None

# sidebar with navigation
sideBar = html.Div(
    [
        html.Img(
            src="data:image/png;base64,{}".format(encoded_image.decode() if encoded_image else ""),
            style={"position": "fixed", "width": "9%",
                   "margin-top": "16px", "display": "block" if encoded_image else "none"},
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

# Page which shows message for mobile device
mobile_page = html.Div(
    [
        html.Img(
            src="data:image/png;base64,{}".format(encoded_image.decode() if encoded_image else ""),
            style={
                "position": "fixed",
                "width": "90%",
                "margin-top": "16px",
                "right": "5%",
                "display": "block" if encoded_image else "none",
            },
        ),
        html.H1(
            "Investment Funnel",
            style={"color": "#ffd0b3", "position": "fixed", "top": "8%", "right": "5%"},
        ),
        html.H4(
            "This page is not available on mobile devices. Please use a desktop browser.",
            style={
                "color": "white",
                "position": "fixed",
                "top": "20%",
                "right": "5%",
                "left": "5%",
            },
        ),
    ],
    style=MOBILE_PAGE,
)
