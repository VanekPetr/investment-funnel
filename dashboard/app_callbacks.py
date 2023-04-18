import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
from models.main import TradeBot
from dash import dcc
from dashboard.app_layouts import page_1_layout, page_2_layout, page_3_layout, page_4_layout

global OPTtable, BENCHtable
global save_Figure
save_Figure = None


OPTtable = pd.DataFrame(np.array([['No result', 'No result', 'No result']]),
                        columns=['Avg An Ret', 'Std Dev of Ret', 'Sharpe R'])
BENCHtable = pd.DataFrame(np.array([['No result', 'No result', 'No result']]),
                          columns=['Avg An Ret', 'Std Dev of Ret', 'Sharpe R'])


algo = TradeBot()


def get_callbacks(app):
    # WHICH WEBPAGE
    @app.callback(
        Output('page-content', 'children'),
        [Input('url', 'pathname')]
    )
    def display_page(pathname):
        if pathname == '/':
            return page_1_layout
        elif pathname == '/page-1':
            return page_2_layout
        elif pathname == '/page-2':
            return page_3_layout
        else:
            return page_4_layout

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
         Output('first-run-page-3-2', 'data'),
         Output('click-prev', 'data')
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
         State('picker-test', 'end_date'),
         State('first-run-page-3-2', 'data'),
         State('click-prev', 'data')]
    )
    def plot_backtest(click, ml_method, num_runs, num_clusters, scen_method, scen_num, benchmark,
                      start_data, end_train, start_test, end_data, first_run_page_3_2, click_prev):
        global OPTtable, BENCHtable
        global save_Figure3, save_Figure3_comp
        global save_ml, save_ml_num, save_clust_top, save_scen, save_scen_num, save_bench

        if first_run_page_3_2 < 1:
            first_run_page_3_2 = 1
            save_Figure3, save_Figure3_comp = None, None
            save_ml, save_ml_num, save_clust_top, save_scen, save_scen_num, save_bench = None, 2, 5, None, 1000, None

            return (save_Figure3, save_Figure3_comp, OPTtable.to_dict('records'), BENCHtable.to_dict('records'),
                    save_ml, save_ml_num, save_clust_top, save_scen, save_scen_num, save_bench, first_run_page_3_2,
                    click_prev)

        if click is None:
            click = click_prev
        elif click < click_prev:
            click = click_prev + 1

        if click > click_prev:
            # RUN ML algo
            if ml_method == 'MST':
                _, subset_of_assets = algo.mst(start_date=start_data, end_date=end_train, n_mst_runs=num_runs)
            else:
                _, subset_of_assets = algo.clustering(start_date=start_data, end_date=end_train, n_clusters=num_runs,
                                                      n_assets=num_clusters)
            # RUN THE BACKTEST
            OPTtable, BENCHtable, figPerf, figComp = algo.backtest(start_train_date=start_data,
                                                                   end_train_date=end_train,
                                                                   start_test_date=start_test,
                                                                   end_test_date=end_data,
                                                                   subset_of_assets=subset_of_assets,
                                                                   benchmarks=benchmark,
                                                                   scenarios_type=scen_method,
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

        return (save_Figure3, save_Figure3_comp, OPTtable.to_dict('records'), BENCHtable.to_dict('records'),
                save_ml, save_ml_num, save_clust_top, save_scen, save_scen_num, save_bench, first_run_page_3_2,
                click_prev)

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
         Output('picker-train', 'end_date'),
         Output('first-run-page-3', 'data')],
        [Input('picker-train', 'end_date')],
        [State('first-run-page-3', 'data')]
    )
    def update_test_date(selected_date, first_run_page_3):
        global minDate, maxDate
        global final_date
        if first_run_page_3 < 1:
            final_date = '2017-07-01'
            first_run_page_3 = 1
        elif selected_date != None:
            final_date = selected_date

        return final_date, maxDate, minDate, maxDate, minDate, maxDate, minDate, final_date, first_run_page_3

    # AI Feature Selection
    # ----------------------------------------------------------------------------------------------------------------------
    # PLOT ML MST GRAPH
    @app.callback(
        [Output('mlFig', 'children'),
         Output('picker-AI', 'start_date'),
         Output('picker-AI', 'end_date'),
         Output('AIResult', 'data'),
         Output('AInumber', 'children'),
         Output('model-dropdown', 'value'),
         Output('ML-num-dropdown', 'value'),
         Output('saved-start-date-page-1', 'data'),
         Output('saved-end-date-page-1', 'data'),
         Output('saved-ai-table', 'data'),
         Output('saved-ml-model', 'data'),
         Output('saved-ml-spec', 'data'),
         Output('saved-ml-text', 'data')
         ],
        [Input('MLRun', 'n_clicks')],
        [State('model-dropdown', 'value'),
         State('ML-num-dropdown', 'value'),
         State('picker-AI', 'start_date'),
         State('picker-AI', 'end_date'),
         State('saved-start-date-page-1', 'data'),
         State('saved-end-date-page-1', 'data'),
         State('saved-ai-table', 'data'),
         State('saved-ml-model', 'data'),
         State('saved-ml-spec', 'data'),
         State('saved-ml-text', 'data')]
    )
    def plot_ml(
        click_ml, model, spec, start, end, saved_start, saved_end, saved_ai_table, saved_model, saved_spec, saved_text
    ):
        global save_Figure2

        if click_ml:
            selected_start = str(start)
            selected_end = str(end)

            # MST
            if model == "MST":
                # RUN THE MINIMUM SPANNING TREE METHOD
                fig, ai_subset = algo.mst(start_date=selected_start, end_date=selected_end, n_mst_runs=spec, plot=True)
                save_Figure2 = dcc.Graph(figure=fig, style={'height': '800px', 'margin': '0%'})

            # CLUSTERING
            else:
                fig, ai_subset = algo.clustering(start_date=selected_start, end_date=selected_end, n_clusters=spec,
                                                 n_assets=10, plot=True)
                save_Figure2 = dcc.Graph(figure=fig, style={'height': '800px', 'margin': '0%'})

            ai_data = algo.get_stat(start_date=selected_start, end_date=selected_end)
            ai_table = ai_data.loc[list(ai_subset), ['Name', 'ISIN', 'Sharpe Ratio', 'Average Annual Returns',
                                                     'Standard Deviation of Returns']]
            # ROUNDING
            ai_table["Standard Deviation of Returns"] = round(ai_table["Standard Deviation of Returns"], 2)
            ai_table["Average Annual Returns"] = round(ai_table["Average Annual Returns"], 2)

            ml_text = 'Number of selected assets: ' + str(len(ai_table))

            return (save_Figure2, selected_start, selected_end, ai_table.to_dict('records'), ml_text, model, spec,
                    selected_start, selected_end, ai_table.to_dict('records'), model, spec, ml_text)
        else:
            return (save_Figure2, saved_start, saved_end, saved_ai_table, saved_text, saved_model, saved_spec,
                    saved_start, saved_end, saved_ai_table, saved_model, saved_spec, saved_text)

    # MARKET OVERVIEW
    # ----------------------------------------------------------------------------------------------------------------------
    # PLOT GRAPH WITH DOTS
    @app.callback(
        [Output('dotsFig', 'children'),
         Output('picker-show', 'start_date'),
         Output('picker-show', 'end_date'),
         Output('find-fund', 'value'),
         Output('saved-start-date-page-0', 'data'),
         Output('saved-end-date-page-0', 'data'),
         Output('saved-find-fund', 'data')],
        [Input('page-content', 'children'),
         Input('show', 'n_clicks')],
        [State('picker-show', 'start_date'),
         State('picker-show', 'end_date'),
         State('find-fund', 'value'),
         State('saved-start-date-page-0', 'data'),
         State('saved-end-date-page-0', 'data'),
         State('saved-find-fund', 'data')]
    )
    def plot_dots(trigger, click, start, end, search, saved_start, saved_end, saved_find_fund):
        global save_Figure

        if click:
            selected_start = str(start)
            selected_end = str(end)

            fig = algo.plot_dots(start_date=selected_start, end_date=selected_end, fund_set=search)
            save_Figure = dcc.Graph(figure=fig,
                                    style={'position': 'absolute',
                                           'right': '0%',
                                           'bottom': '0%',
                                           'top': '0%',
                                           'left': '0%'}
                                    )
            return save_Figure, selected_start, selected_end, search, selected_start, selected_end, search
        else:
            return save_Figure, saved_start, saved_end, saved_find_fund, saved_start, saved_end, saved_find_fund
