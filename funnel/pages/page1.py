import dash
from dash import html
from ifunnel.models.main import initialize_bot

dash.register_page(__name__, path="/page-1", name="Page 1")

algo = initialize_bot()

layout = html.Div([
    html.H2("Page 1 – Analysis"),
    html.P("Backtest results or strategy summary here."),
    # You could also import and insert `page_1_layout(algo)` here
])
