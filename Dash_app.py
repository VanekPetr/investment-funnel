import dash  # version 1.13.1
import dash_auth
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
from Dash_layouts import *
from models.main import TradeBot

global df_etim
global first_run_page1, first_run_page2, first_run_page3, first_run_page3_2
global ML_click_prev, click_prev
global AItable, OPTtable, BENCHtable

AItable = pd.DataFrame(np.array([['No result', 'No result', 'No result', 'No result', 'No result']]),
                       columns=['Name', 'ISIN', 'Sharpe Ratio', 'Average Annual Returns', 'Standard Deviation of Returns'])
OPTtable = pd.DataFrame(np.array([['No result', 'No result', 'No result']]),
                        columns=['Avg An Ret', 'Std Dev of Ret', 'Sharpe R'])
BENCHtable = pd.DataFrame(np.array([['No result', 'No result', 'No result']]),
                          columns=['Avg An Ret', 'Std Dev of Ret', 'Sharpe R'])
first_run_page1, first_run_page2, first_run_page3, first_run_page3_2 = 0, 0, 0, 0
ML_click_prev, click_prev = 0, 0

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
    'PFO2022': 'student',
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
     Output('tableResult-benchmark', 'data'),
     Output('select-ml', 'value'),
     Output('slider-backtest-ml', 'value'),
     Output('slider-backtest', 'value'),
     Output('select-scenarios', 'value'),
     Output('my-slider2', 'value'),
     Output('select-benchmark', 'value'),
     ],
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
     State('picker-test', 'end_date')]
)
def plot_backtest(click, ml_method, num_runs, num_clusters, scen_method, scen_num, benchmark,
                  start_data, end_train, start_test, end_data):
    global algo
    global first_run_page3_2, click_prev
    global OPTtable, BENCHtable
    global save_Figure3, save_Figure3_comp
    global save_ml, save_ml_num, save_clust_top, save_scen, save_scen_num, save_bench

    if first_run_page3_2 < 1:
        first_run_page3_2 = 1
        save_Figure3, save_Figure3_comp = None, None
        save_ml, save_ml_num, save_clust_top, save_scen, save_scen_num, save_bench = None, 2, 5, None, 1000, None

        return save_Figure3, save_Figure3_comp, OPTtable.to_dict('records'), BENCHtable.to_dict('records'), \
               save_ml, save_ml_num, save_clust_top, save_scen, save_scen_num, save_bench

    if click is None:
        click = click_prev
    elif click < click_prev:
        click = click_prev + 1

    if click > click_prev:
        # SETUP WORKING DATASET, DIVIDE DATASET INTO TRAINING AND TESTING PART?
        algo.setup_data(start=start_data, end=end_data, train_test=True, end_train=end_train, start_test=start_test)
        # RUN ML algo
        if ml_method == 'MST':
            algo.mst(n_mst_runs=num_runs, plot=False)
        else:
            algo.clustering(n_clusters=num_runs, n_assets=num_clusters, plot=False)
        # RUN THE BACKTEST
        OPTtable, BENCHtable, figPerf, figComp = algo.backtest(assets=ml_method,
                                                               benchmark=benchmark,
                                                               scenarios=scen_method,
                                                               n_simulations=scen_num)
        # Save page values
        save_Figure3 = dcc.Graph(figure=figPerf, style={'margin': '0%'})
        save_Figure3_comp = dcc.Graph(figure=figComp, style={'margin': '0%'})
        save_ml = ml_method
        save_ml_num = num_runs
        save_clust_top = num_clusters
        save_scen = scen_method
        save_scen_num = scen_num
        save_bench = benchmark

    return save_Figure3, save_Figure3_comp, OPTtable.to_dict('records'), BENCHtable.to_dict('records'), \
           save_ml, save_ml_num, save_clust_top, save_scen, save_scen_num, save_bench


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
     Output('AInumber', 'children'),
     Output('model-dropdown', 'value'),
     Output('ML-num-dropdown', 'value')
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
    global first_run_page2, ML_click_prev
    global save_Figure2, AItable, AI_text_number, save_model, save_MLnum

    if first_run_page2 < 1:
        first_run_page2 = 1
        startDate2 = startDate
        endDate2 = endDate
        save_Figure2, save_model, save_MLnum = None, None, None
        AI_text_number = "No selected asset."
        return save_Figure2, startDate, endDate, minDate, maxDate, AItable.to_dict('records'), AI_text_number,\
               save_model, save_MLnum

    if click_ML is None:
        click_ML = ML_click_prev
    elif click_ML < ML_click_prev:
        click_ML = ML_click_prev + 1

    if click_ML > ML_click_prev:
        startDate2 = start
        endDate2 = end
        save_model = model
        save_MLnum = num_iter
        # SETUP WORKING DATASET, DIVIDE DATASET INTO TRAINING AND TESTING PART?
        algo.setup_data(start=startDate2, end=endDate2, train_test=False)
        # MST
        if model == "MST":
            # RUN THE MINIMUM SPANNING TREE METHOD
            fig = algo.mst(n_mst_runs=num_iter, plot=True)
            AIsubset = algo.subsetMST
            ML_click_prev = click_ML
            save_Figure2 = dcc.Graph(figure=fig, style={'height': '800px', 'margin': '0%'})
        # CLUSTERING
        else:
            fig = algo.clustering(n_clusters=num_iter, n_assets=10, plot=True)
            AIsubset = algo.subsetCLUST
            save_Figure2 = dcc.Graph(figure=fig, style={'height': '800px', 'margin': '0%'})
        AItable = algo.AIdata.loc[list(AIsubset), ['Name', 'ISIN', 'Sharpe Ratio', 'Average Annual Returns',
                                                   'Standard Deviation of Returns']]
        # ROUNDING
        AItable["Standard Deviation of Returns"] = round(AItable["Standard Deviation of Returns"], 2)
        AItable["Average Annual Returns"] = round(AItable["Average Annual Returns"], 2)

        AI_text_number = 'Number of selected assets: ' + str(len(AItable))

    return save_Figure2, startDate2, endDate2, minDate, maxDate, AItable.to_dict('records'), AI_text_number,\
           save_model, save_MLnum


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

            fig = algo.plot_dots(start=str(start), end=str(end), fund_set=save_Search)
            save_Figure = dcc.Graph(figure=fig, style={'position': 'absolute', 'right': '0%', 'bottom': '0%',
                                                       'top': '0%', 'left': '0%'})
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
    # TODO!! change log same as for project avocado
