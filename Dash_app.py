global df_etim
import dash  # version 1.13.1
import dash_auth
import numpy as np
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

global first_run_page1, first_run_page2, first_run_page3
global ML_click_prev, AItable
ML_click_prev = 0
AItable = pd.DataFrame(np.array([['No result', 'No result', 'No result', 'No result', 'No result']]),
                       columns=['Name', 'ISIN', 'Sharpe Ratio', 'Average Annual Returns', 'Standard Deviation of Returns'])
first_run_page1, first_run_page2, first_run_page3 = 0, 0, 0

'''
# ----------------------------------------------------------------------------------------------------------------------
# APP
# ----------------------------------------------------------------------------------------------------------------------
'''
# Name and password
VALID_USERNAME_PASSWORD_PAIRS = {
    'Petr': 'algo94',
    'CHY': 'algo123',
    'PFO2021': 'student',
    'Trader': '42'
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
     Output('tableResult', 'data'),
     Output('tableResult-benchmark', 'data')],
    [Input('backtestRun', 'n_clicks')],
    [State('select-ml', 'value'),
     State('slider-backtest-ml', 'value'),
     State('slider-backtest', 'value'),
     State('select-scenarios', 'value'),
     State('my-slider2', 'value'),
     State('select-benchmark', 'value'),
     State('picker-train', 'start_date'),
     State('picker-train', 'end_date'),
     State('picker-test', 'start_date'),
     State('picker-test', 'end_date')],
    prevent_initial_call=True
)
def plot_backtest(click, ml_method, num_runs, num_clusters, scen_method, scen_num, benchmark,
                  start_data, end_train, start_test, end_data):
    global algo

    if click > 0:
        # SETUP WORKING DATASET, DIVIDE DATASET INTO TRAINING AND TESTING PART?
        algo.setup_data(start=start_data, end=end_data, train_test=True, end_train=end_train, start_test=start_test)
        # RUN ML algo
        if ml_method == 'MST':
            algo.mst(nMST=num_runs, plot=False)
        else:
            algo.clustering(nClusters=num_runs, nAssets=num_clusters, plot=False)
        # RUN THE BACKTEST
        results, results_benchmark, figPerf, figComp = algo.backtest(assets=ml_method,
                                                                     benchmark=benchmark,
                                                                     scenarios=scen_method,
                                                                     nSimulations=scen_num,
                                                                     plot=True)

    return dcc.Graph(figure=figPerf, style={'margin':'0%'}),\
           dcc.Graph(figure=figComp, style={'margin':'0%'}), \
           results.to_dict('records'), results_benchmark.to_dict('records')

@app.callback(
    Output('slider-output-container2', 'children'),
    [Input('my-slider2', 'value')])
def update_output(value):
    return '# of scenarios: {}'.format(value)

@app.callback(
    Output('slider-output-container-backtest', 'children'),
    [Input('slider-backtest', 'value')])
def update_output_cluster(value):
    return 'In case of CLUSTERING: # of the best performing assets selected from each cluster: {}'.format(value)

@app.callback(
    Output('slider-output-container-backtest-ml', 'children'),
    [Input('slider-backtest-ml', 'value')])
def update_output_MLtype(value):
    return '# of clusters or # of MST runs: {}'.format(value)

@app.callback(
    [Output('picker-test', 'start_date'),
     Output('picker-test', 'end_date'),
     Output('picker-test', 'min_date_allowed'),
     Output('picker-test', 'max_date_allowed'),
     Output('picker-train', 'min_date_allowed'),
     Output('picker-train', 'max_date_allowed'),
     Output('picker-train', 'start_date'),
     Output('picker-train', 'end_date')],
    [Input('picker-train', 'end_date')])
def update_test_date(selected_date):
    global first_run_page3
    global minDate, maxDate
    global final_date
    if first_run_page3 < 1:
        final_date = '2017-07-01'
        first_run_page3 = 1
    elif selected_date != None:
        final_date = selected_date

    return final_date, maxDate, minDate, maxDate, minDate, maxDate, minDate, final_date


# AI Feature Selection
# ----------------------------------------------------------------------------------------------------------------------

# PLOT ML MST GRAPH
@app.callback(
    [Output('mlFig', 'children'),
     Output('picker-AI', 'start_date'),
     Output('picker-AI', 'end_date'),
     Output('picker-AI', 'min_date_allowed'),
     Output('picker-AI', 'max_date_allowed'),
     Output('AIResult', 'data'),
     Output('AInumber', 'children')
     ],
    [Input('MLRun', 'n_clicks')],
    [State('model-dropdown', 'value'),
     State('ML-num-dropdown', 'value'),
     State('picker-AI', 'start_date'),
     State('picker-AI', 'end_date'),
     ]
)
def plot_ml(click_ML, model, num_iter, start, end):
    global algo
    global startDate, endDate, startDate2, endDate2
    global minDate, maxDate
    global first_run_page2
    global ML_click_prev, clust_click_prev
    global save_Figure2, AItable, AI_text_number

    if first_run_page2 < 1:
        first_run_page2 = 1
        startDate2 = startDate
        endDate2 = endDate
        save_Figure2 = None
        AI_text_number = "No selected asset."
        return save_Figure2, startDate, endDate, minDate, maxDate, AItable.to_dict('records'), AI_text_number

    if click_ML is None:
        click_ML = ML_click_prev
    elif click_ML < ML_click_prev:
        click_ML = ML_click_prev + 1

    if click_ML > ML_click_prev:
        startDate2 = start
        endDate2 = end
        # SETUP WORKING DATASET, DIVIDE DATASET INTO TRAINING AND TESTING PART?
        algo.setup_data(start=startDate2, end=endDate2, train_test=False)
        # MST
        if model == "MST":
            # RUN THE MINIMUM SPANNING TREE METHOD
            fig = algo.mst(nMST=num_iter, plot=True)
            AIsubset = algo.subsetMST
            ML_click_prev = click_ML
            save_Figure2 = dcc.Graph(figure=fig, style={'height':'800px', 'margin':'0%'})
        # CLUSTERING
        else:
            fig = algo.clustering(nClusters=num_iter, nAssets=10, plot=True)
            AIsubset = algo.subsetCLUST
            save_Figure2 = dcc.Graph(figure=fig, style={'height':'800px', 'margin':'0%'})
        AItable = algo.AIdata.loc[list(AIsubset), ['Name', 'ISIN', 'Sharpe Ratio', 'Average Annual Returns',
                                                   'Standard Deviation of Returns']]
        # ROUNDING
        AItable["Standard Deviation of Returns"] = round(AItable["Standard Deviation of Returns"], 2)
        AItable["Average Annual Returns"] = round(AItable["Average Annual Returns"], 2)

        AI_text_number = 'Number of selected assets is: ' + str(len(AItable))

    return save_Figure2, startDate2, endDate2, minDate, maxDate, AItable.to_dict('records'), AI_text_number

# MARKET OVERVIEW
# ----------------------------------------------------------------------------------------------------------------------

# PLOT GRAPH WITH DOTS
@app.callback(
    [Output('dotsFig', 'children'),
     Output('picker-show', 'start_date'),
     Output('picker-show', 'end_date'),
     Output('picker-show', 'min_date_allowed'),
     Output('picker-show', 'max_date_allowed'),
     Output('find-fund', 'value')
     ],
    [Input('show', 'n_clicks')],
    [State('picker-show', 'start_date'),
     State('picker-show', 'end_date'),
     State('find-fund', 'value')],
)
def plot_dots(click, start, end, search):
    global algo
    global startDate, endDate
    global minDate, maxDate
    global first_run_page1, save_Figure, save_Search

    if first_run_page1 < 1:
        algo = TradeBot()
        startDate = algo.weeklyReturns.index[0]
        endDate = algo.weeklyReturns.index[-2]
        minDate = algo.weeklyReturns.index[0]
        maxDate = algo.weeklyReturns.index[-2]

        save_Figure = None
        save_Search = []
        first_run_page1 = 1

        return save_Figure, startDate, endDate, minDate, maxDate, save_Search

    try:
        if click > 0:
            startDate = start
            endDate = end
            save_Search = search

            fig = algo.plot_dots(start=str(start), end=str(end), fundSet=save_Search)
            save_Figure = dcc.Graph(figure=fig, style={'position': 'absolute', 'right': '0%', 'bottom': '0%', 'top': '0%',
                                                       'left': '0%'})
            return save_Figure, startDate, endDate, minDate, maxDate, save_Search
    except:
        return save_Figure, startDate, endDate, minDate, maxDate, save_Search

'''
# ----------------------------------------------------------------------------------------------------------------------
# RUN!
# ----------------------------------------------------------------------------------------------------------------------
'''
if __name__ == '__main__':
    app.run_server(debug=False, dev_tools_hot_reload=False)
