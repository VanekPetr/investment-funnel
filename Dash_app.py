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

global first_run_page1, first_run_page2
global mst_click_prev, clust_click_prev
mst_click_prev, clust_click_prev = 0, 0
first_run_page1, first_run_page2 = 0, 0

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
     State('slider-backtest-ml', 'value'),
     State('slider-backtest', 'value'),
     State('select-scenarios', 'value'),
     State('my-slider2', 'value'),
     State('select-benchmark', 'value')],
    prevent_initial_call=True
)
def plot_backtest(click, ml_method, num_runs, num_clusters, scen_method, scen_num, benchmark):
    global algo
    global startDate, endDate

    if click > 0:
        # SETUP WORKING DATASET, DIVIDE DATASET INTO TRAINING AND TESTING PART?
        algo.setup_data(start=startDate, end=endDate, train_test=True, train_ratio=0.6)
        # RUN ML algo
        if ml_method == 'MST':
            algo.mst(nMST=num_runs, plot=False)
        else:
            algo.clustering(nClusters=num_runs, nAssets=num_clusters, plot=False)
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

@app.callback(
    Output('slider-output-container-backtest', 'children'),
    [Input('slider-backtest', 'value')])
def update_output_cluster(value):
    return 'In case of CLUSTERING: Number of the best performing assets selected from each cluster is {}'.format(value)

@app.callback(
    Output('slider-output-container-backtest-ml', 'children'),
    [Input('slider-backtest-ml', 'value')])
def update_output_MLtype(value):
    return '# of clusters or # of MST runs: {}'.format(value)


# AI Feature Selection
# ----------------------------------------------------------------------------------------------------------------------

# PLOT ML GRAPH
@app.callback(
    [Output('mlFig', 'children'),
     Output('picker-AI', 'start_date'),
     Output('picker-AI', 'end_date'),
     Output('picker-AI', 'min_date_allowed'),
     Output('picker-AI', 'max_date_allowed')
     ],
    [Input('mstRun', 'n_clicks'),
     Input('clusterRun', 'n_clicks')],
    [State('mst-dropdown', 'value'),
     State('cluster-dropdown', 'value'),
     State('picker-AI', 'start_date'),
     State('picker-AI', 'end_date'),
     ]
)
def plot_ml(click_mst, click_clust, mst, clust, start, end):
    global algo
    global startDate, endDate, startDate2, endDate2
    global minDate, maxDate
    global first_run_page2
    global mst_click_prev, clust_click_prev
    global save_Figure2

    if first_run_page2 < 1:
        first_run_page2 = 1
        startDate2 = startDate
        endDate2 = endDate
        save_Figure2 = None
        return save_Figure2, startDate, endDate, minDate, maxDate

    if click_mst is None:
        click_mst = mst_click_prev
    elif click_mst < mst_click_prev:
        click_mst = mst_click_prev + 1

    if click_clust is None:
        click_clust = clust_click_prev
    elif click_clust < clust_click_prev:
        click_clust = clust_click_prev + 1

    if click_mst > mst_click_prev:
        startDate2 = start
        endDate2 = end
        # SETUP WORKING DATASET, DIVIDE DATASET INTO TRAINING AND TESTING PART?
        algo.setup_data(start=startDate2, end=endDate2, train_test=False)
        # RUN THE MINIMUM SPANNING TREE METHOD
        fig = algo.mst(nMST=mst, plot=True)
        mst_click_prev = click_mst
        save_Figure2 = dcc.Graph(figure=fig, style={'position': 'absolute', 'right': '0%', 'bottom': '0%', 'top': '0%',
                                                    'left': '0%'})

    elif click_clust > clust_click_prev:
        startDate2 = start
        endDate2 = end
        # SETUP WORKING DATASET, DIVIDE DATASET INTO TRAINING AND TESTING PART?
        algo.setup_data(start=startDate2, end=endDate2, train_test=False)
        fig = algo.clustering(nClusters=clust, nAssets=2, plot=True)
        save_Figure2 = dcc.Graph(figure=fig, style={'position': 'absolute', 'right': '0%', 'bottom': '0%', 'top': '0%',
                                                    'left': '0%'})

    return save_Figure2, startDate2, endDate2, minDate, maxDate


# MARKET OVERVIEW
# ----------------------------------------------------------------------------------------------------------------------

# PLOT GRAPH WITH DOTS
@app.callback(
    [Output('dotsFig', 'children'),
     Output('picker-show', 'start_date'),
     Output('picker-show', 'end_date'),
     Output('picker-show', 'min_date_allowed'),
     Output('picker-show', 'max_date_allowed')
     ],
    [Input('show', 'n_clicks')],
    [State('picker-show', 'start_date'),
     State('picker-show', 'end_date')],
)
def plot_dots(click, start, end):
    global algo
    global startDate, endDate
    global minDate, maxDate
    global first_run_page1, save_Figure

    if first_run_page1 < 1:
        algo = TradeBot()
        startDate = algo.weeklyReturns.index[0]
        endDate = algo.weeklyReturns.index[-2]
        minDate = algo.weeklyReturns.index[0]
        maxDate = algo.weeklyReturns.index[-2]

        save_Figure = None
        first_run_page1 = 1

        return save_Figure, startDate, endDate, minDate, maxDate

    try:
        if click > 0:
            startDate = start
            endDate = end

            fig = algo.plot_dots(start=str(start), end=str(end))
            save_Figure = dcc.Graph(figure=fig, style={'position': 'absolute', 'right': '0%', 'bottom': '0%', 'top': '0%',
                                                       'left': '0%'})
            return save_Figure, startDate, endDate, minDate, maxDate
    except:
        return save_Figure, startDate, endDate, minDate, maxDate



'''
# ----------------------------------------------------------------------------------------------------------------------
# RUN!
# ----------------------------------------------------------------------------------------------------------------------
'''
if __name__ == '__main__':
    app.run_server(debug=False, dev_tools_hot_reload=False)
