import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dashboard.app_layouts import page_1_layout
from dashboard.app_callbacks import get_callbacks


def load_page():
    return html.Div([dcc.Location(id='url'), html.Div(id='page-content', children=page_1_layout)])


# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = load_page()
# App callbacks
get_callbacks(app)


if __name__ == '__main__':
    app.run_server(debug=False, dev_tools_hot_reload=False)
