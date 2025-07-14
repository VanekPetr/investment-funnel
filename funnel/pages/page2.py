import dash
from dash import html

dash.register_page(__name__, path="/page-2", name="Page 2")

layout = html.Div([
    html.H2("Page 2 – Fund Selection"),
    html.P("This is another module of the dashboard.")
])
