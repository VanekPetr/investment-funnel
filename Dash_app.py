global df_etim
import dash  # version 1.13.1
import dash_auth
from dash.dependencies import Input, Output, State
from dash_extensions.snippets import send_data_frame
from dash.exceptions import PreventUpdate
import plotly.express as px
import datetime
import os
from urllib.parse import quote as urlquote
from Dash_layouts import *
from dash_extensions.snippets import send_file
from main import TradeBot
from dataAnalyser import tickers

global temp_storage
global Prev_save_click
global prev_new_row_click
global just_retrained
prev_new_row_click = 0
Prev_save_click = 0
temp_storage = 0
temp_click = 1
temp2_click = 0
temp_name = ''
just_retrained = False

global mst_click_prev, clust_click_prev
mst_click_prev = 0
clust_click_prev = 0

'''
# ----------------------------------------------------------------------------------------------------------------------
# APP
# ----------------------------------------------------------------------------------------------------------------------
'''
# Name and password
VALID_USERNAME_PASSWORD_PAIRS = {
    'Petr': 'algo94',
    'CHY': 'algo123'
}
# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Run app on server
server = app.server

# Authentication
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)


def load_page():
    return html.Div([
        dcc.Location(id='url'),
        html.Div(id='page-content', children=page_1_layout)
    ])


# APP layout
app.layout = load_page()

'''
# ----------------------------------------------------------------------------------------------------------------------
# CALLBACKS AND FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
'''

# WHICH WEBPAGE
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    global df_etim
    global df_data
    if pathname == '/':
        return page_1_layout
    elif pathname == '/page-1':
        return page_2_layout
    else:
        return page_3_layout


# BACK-TESTING
# ----------------------------------------------------------------------------------------------------------------------

# PLOT GRAPH WITH DOTS
@app.callback(
    [Output('backtestPerfFig', 'children'),
     Output('backtestCompFig', 'children'),
     Output('tableResult', 'data')],
    [Input('backtestRun', 'n_clicks')],
    [State('select-ml', 'value'),
     State('select-scenarios', 'value'),
     State('my-slider2', 'value'),
     State('select-benchmark', 'value')],
    prevent_initial_call=True
)
def plot_backtest(click, ml_method, scen_method, scen_num, benchmark):
    global algo

    if click > 0:
        # RUN THE BACKTEST
        results, figPerf, figComp = algo.backtest(assets=ml_method,
                                                  benchmark=benchmark,
                                                  scenarios=scen_method,
                                                  nSimulations=scen_num,
                                                  plot=True)

    return dcc.Graph(figure=figPerf, style={'position': 'absolute', 'right': '0%', 'bottom': '0%', 'top': '0%',
                                        'left': '0%'}),\
           dcc.Graph(figure=figComp, style={'position': 'absolute', 'right': '0%', 'bottom': '0%', 'top': '0%',
                                        'left': '0%'}), \
           results.to_dict('records')

@app.callback(
    Output('slider-output-container2', 'children'),
    [Input('my-slider2', 'value')])
def update_output(value):
    return 'Selected number of scenarios: {}'.format(value)


# AI Feature Selection
# ----------------------------------------------------------------------------------------------------------------------

# PLOT ML GRAPH
@app.callback(
    Output('mlFig', 'children'),
    [Input('mstRun', 'n_clicks'),
     Input('clusterRun', 'n_clicks')],
    [State('mst-dropdown', 'value'),
     State('cluster-dropdown', 'value'),
     State('my-slider', 'value')],
    prevent_initial_call=True
)
def plot_ml(click_mst, click_clust, mst, clust, clust_num):
    global algo
    global startDate, endDate
    global mst_click_prev, clust_click_prev

    if click_mst is None:
        click_mst = mst_click_prev
    elif click_mst < mst_click_prev:
        click_mst = mst_click_prev + 1

    if click_clust is None:
        click_clust = clust_click_prev
    elif click_clust < clust_click_prev:
        click_clust = clust_click_prev + 1

    # SETUP WORKING DATASET, DIVIDE DATASET INTO TRAINING AND TESTING PART?
    algo.setup_data(start=startDate, end=endDate, train_test=True, train_ratio=0.6)

    if click_mst > mst_click_prev:
        # RUN THE MINIMUM SPANNING TREE METHOD
        fig = algo.mst(nMST=mst, plot=True)
        mst_click_prev = click_mst

    elif click_clust > clust_click_prev:
        fig = algo.clustering(nClusters=clust, nAssets=clust_num, plot=True)

    return dcc.Graph(figure=fig,
                     style={'position': 'absolute', 'right': '0%', 'bottom': '0%', 'top': '0%', 'left': '0%'})


@app.callback(
    Output('slider-output-container', 'children'),
    [Input('my-slider', 'value')])
def update_output(value):
    return 'Number of the best performing assets selected from each cluster: {}'.format(value)


# MARKET OVERVIEW
# ----------------------------------------------------------------------------------------------------------------------

# PLOT GRAPH WITH DOTS
@app.callback(
    Output('dotsFig', 'children'),
    [Input('show', 'n_clicks')],
    [State('picker-show', 'start_date'),
     State('picker-show', 'end_date')],
    prevent_initial_call=True
)
def plot_dots(click, start, end):
    global algo
    global startDate, endDate

    if click > 0:
        startDate = start
        endDate = end

        fig = algo.plot_dots(start=str(start), end=str(end))

    return dcc.Graph(figure=fig, style={'position': 'absolute', 'right': '0%', 'bottom': '0%', 'top': '0%',
                                        'left': '0%'})


# DOWNLOAD THE DATA
@app.callback(
    [Output('picker-show', 'min_date_allowed'),
     Output('picker-show', 'max_date_allowed'),
     Output('picker-show', 'start_date'),
     Output('picker-show', 'end_date')],
    Input('download', 'n_clicks'),
    [State('picker-download', 'start_date'),
     State('picker-download', 'end_date')],
    prevent_initial_call=True
)
def download_data(click, start, end):
    global algo

    if click > 0:
        algo = TradeBot(start=str(start), end=str(end), assets=tickers)

    return start, end, start, end


'''
# ----------------------------------------------------------------------------------------------------------------------
# RUN!
# ----------------------------------------------------------------------------------------------------------------------
'''
if __name__ == '__main__':
    app.run_server(debug=False, dev_tools_hot_reload=False)
