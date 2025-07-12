import dash_bootstrap_components as dbc
from dash import html

from .general import encoded_image
from .styles import MOBILE_PAGE


def create_mobile_layout(algo):
    """Create the layout for the mobile page."""
    return _divs(algo)


def _divs(algo) -> html.Div:
    # Page which shows message for mobile device
    image = encoded_image()
    mobile_page = html.Div(
        [
            html.Img(
                src="data:image/png;base64,{}".format(image.decode() if image else ""),
                style={
                    "position": "fixed",
                    "width": "90%",
                    "margin-top": "16px",
                    "right": "5%",
                    "display": "block" if image else "none",
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
