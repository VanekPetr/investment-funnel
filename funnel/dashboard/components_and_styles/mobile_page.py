import base64
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import html

from .styles import (
    MOBILE_PAGE,
)

encoded_image = base64.b64encode(
    open(Path(__file__).parent / "assets" / "ALGO_logo.png", "rb").read()
)

def div(algo):
    """
    Create the layout for the mobile page.

    This function creates the layout for the mobile page, which displays a message
    indicating that the application is not available on mobile devices.

    Args:
        algo: The algorithm object containing data and methods (not used in this function)

    Returns:
        html.Div: The layout for the mobile page
    """
    # Page which shows message for mobile device
    mobile_page = html.Div(
        [
            html.Img(
                src="data:image/png;base64,{}".format(encoded_image.decode()),
                style={
                    "position": "fixed",
                    "width": "90%",
                    "margin-top": "16px",
                    "right": "5%",
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

    return html.Div(
        [
            # Row 1 - body
            dbc.Row([mobile_page])
        ]
    )
