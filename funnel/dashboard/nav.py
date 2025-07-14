from dash import html
import dash_bootstrap_components as dbc


def get_navbar():
    return dbc.Col(
        [
            # Logo (replace with actual image if you have one)
            html.Div(
                html.Img(
                    src="/assets/logo.png",  # Make sure you have this file in `assets/`
                    style={"width": "100%", "padding": "1rem"}
                ),
                style={"textAlign": "center"}
            ),
            # Navigation links
            html.Div(
                [
                    dbc.Nav(
                        [
                            dbc.NavLink("Page 1", href="/page-1", active="exact"),
                            dbc.NavLink("Page 2", href="/page-2", active="exact"),
                            dbc.NavLink("Overview", href="/overview", active="exact"),
                        ],
                        vertical=True,
                        pills=True,
                    )
                ],
                style={"padding": "1rem"}
            )
        ],
        width=2,
        style={
            "backgroundColor": "#f8f9fa",
            "height": "100vh",
            "position": "fixed",
            "left": 0,
            "top": 0,
            "bottom": 0,
            "paddingTop": "1rem",
            "borderRight": "1px solid #dee2e6"
        }
    )