import flask
from dash import dcc
from dash.dependencies import Input, Output, State

from ..models.main import TradeBot
from .app_layouts import (
    page_1_layout,
    page_2_layout,
    page_3_layout,
    page_4_layout,
    page_mobile_layout,
)

algo = TradeBot()


def get_callbacks(app):
    # WHICH WEBPAGE
    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def display_page(pathname: str):
        is_mobile = flask.request.headers.get("User-Agent").lower()
        if "mobile" in is_mobile or "mobi" in is_mobile:
            return page_mobile_layout
        elif pathname == "/":
            return page_1_layout
        elif pathname == "/page-1":
            return page_2_layout
        elif pathname == "/page-2":
            return page_3_layout
        else:
            return page_4_layout

    # BACK-TESTING
    # -----------------------------------------------------------------------------------------------------------------
    @app.callback(
        [
            Output("glidepaths-output-fig", "children"),
            Output("performance-output-fig", "children"),
            Output("lifecycle-all-output-fig", "children"),
            Output("select-ml-lifecycle", "value"),
            Output("slider-lifecycle-ml", "value"),
            Output("slider-lifecycle", "value"),
            Output("select-scenarios-lifecycle", "value"),
            Output("my-slider-2-lifecycle", "value"),
            Output("saved-ml-model-lifecycle", "data"),
            Output("saved-ml-spec-lifecycle", "data"),
            Output("saved-pick-num-lifecycle", "data"),
            Output("saved-scen-model-lifecycle", "data"),
            Output("saved-scen-spec-lifecycle", "data"),
            Output("saved_portfolio_value", "data"),
            Output("saved_yearly_withdraws", "data"),
            Output("saved_risk_preference", "data"),
            Output("saved_end_year", "data"),
            Output("saved-glidepaths-figure-page-3", "data"),
            Output("saved-performance-figure-page-3", "data"),
            Output("saved-lifecycle-all-figure-page-3", "data"),
            Output("loading-output-lifecycle", "children"),
        ],
        Input("lifecycle-run", "n_clicks"),
        [
            State("select-ml-lifecycle", "value"),
            State("slider-lifecycle-ml", "value"),
            State("slider-lifecycle", "value"),
            State("select-scenarios-lifecycle", "value"),
            State("my-slider-2-lifecycle", "value"),
            State("picker-lifecycle", "start_date"),
            State("picker-lifecycle", "end_date"),
            State("slider-final-year-lifecycle", "value"),
            State("initial-portfolio-value-lifecycle", "value"),
            State("yearly-withdraws-lifecycle", "value"),
            State("initial-risk-appetite-lifecycle", "value"),
            State("saved-ml-model-lifecycle", "data"),
            State("saved-ml-spec-lifecycle", "data"),
            State("saved-pick-num-lifecycle", "data"),
            State("saved-scen-model-lifecycle", "data"),
            State("saved-scen-spec-lifecycle", "data"),
            State("saved_portfolio_value", "data"),
            State("saved_yearly_withdraws", "data"),
            State("saved_risk_preference", "data"),
            State("saved_end_year", "data"),
            State("saved-glidepaths-figure-page-3", "data"),
            State("saved-performance-figure-page-3", "data"),
            State("saved-lifecycle-all-figure-page-3", "data"),
        ],
    )
    def plot_lifecycle(
        click,
        model,
        model_spec,
        pick_top,
        scen_model,
        scen_spec,
        start_data,
        end_train,
        end_year,
        portfolio_value,
        yearly_withdraws,
        risk_preference,
        saved_model,
        saved_model_spec,
        saved_pick_top,
        saved_scen_model,
        saved_scen_spec,
        saved_portfolio_value,
        saved_yearly_withdraws,
        saved_risk_preference,
        saved_end_year,
        saved_glidepaths_figure,
        saved_performance_figure,
        saved_lifecycle_all_figure,
    ):
        # Lifecycle analysis
        if click:
            # RUN ML algo
            if model == "MST":
                _, subset_of_assets = algo.mst(
                    start_date=start_data, end_date=end_train, n_mst_runs=model_spec
                )
            else:
                _, subset_of_assets = algo.clustering(
                    start_date=start_data,
                    end_date=end_train,
                    n_clusters=model_spec,
                    n_assets=pick_top,
                )
            # RUN THE LIFECYCLE FUNCTION
            _, _, fig_performance, fig_glidepaths, _, _, fig_composition_all = (
                algo.lifecycle_scenario_analysis(
                    subset_of_assets=subset_of_assets,
                    scenarios_type=scen_model,
                    n_simulations=scen_spec,
                    end_year=end_year,
                    withdrawals=yearly_withdraws,
                    initial_risk_appetite=risk_preference / 100,
                    initial_budget=portfolio_value,
                )
            )

            performance_figure = dcc.Graph(
                figure=fig_performance, style={"margin": "0%", "height": "800px"}
            )
            glidepaths_figure = dcc.Graph(
                figure=fig_glidepaths, style={"margin": "0%", "height": "800px"}
            )
            lifecycle_all_figure = dcc.Graph(
                figure=fig_composition_all, style={"margin": "0%", "height": "1300px"}
            )

            return (
                glidepaths_figure,
                performance_figure,
                lifecycle_all_figure,
                model,
                model_spec,
                pick_top,
                scen_model,
                scen_spec,
                model,
                model_spec,
                pick_top,
                scen_model,
                scen_spec,
                end_year,
                portfolio_value,
                yearly_withdraws,
                risk_preference,
                glidepaths_figure,
                performance_figure,
                lifecycle_all_figure,
                True,
            )
        else:
            return (
                saved_glidepaths_figure,
                saved_performance_figure,
                saved_lifecycle_all_figure,
                saved_model,
                saved_model_spec,
                saved_pick_top,
                saved_scen_model,
                saved_scen_spec,
                saved_model,
                saved_model_spec,
                saved_pick_top,
                saved_scen_model,
                saved_scen_spec,
                saved_portfolio_value,
                saved_yearly_withdraws,
                saved_risk_preference,
                saved_end_year,
                saved_glidepaths_figure,
                saved_performance_figure,
                saved_lifecycle_all_figure,
                True,
            )

    # BACK-TESTING
    # -----------------------------------------------------------------------------------------------------------------
    @app.callback(
        [
            Output("backtestPerfFig", "children"),
            Output("backtestCompFig", "children"),
            Output("select-ml", "value"),
            Output("slider-backtest-ml", "value"),
            Output("slider-backtest", "value"),
            Output("select-scenarios", "value"),
            Output("my-slider2", "value"),
            Output("select-benchmark", "value"),
            Output("saved-ml-model-back", "data"),
            Output("saved-ml-spec-back", "data"),
            Output("saved-pick-num-back", "data"),
            Output("saved-scen-model-back", "data"),
            Output("saved-scen-spec-back", "data"),
            Output("saved-benchmark-back", "data"),
            Output("saved-perf-figure-page-2", "data"),
            Output("saved-comp-figure-page-2", "data"),
            Output("loading-output-backtest", "children"),
            Output("backtestUniverseFig", "children"),
            Output("saved-universe-figure-page-2", "data"),
            Output("select-solver", "value"),
            Output("saved-solver", "data"),
            Output("select-optimization-model", "value"),
            Output("saved-optimization-model", "data"),
        ],
        Input("backtestRun", "n_clicks"),
        [
            State("select-ml", "value"),
            State("slider-backtest-ml", "value"),
            State("slider-backtest", "value"),
            State("select-scenarios", "value"),
            State("my-slider2", "value"),
            State("select-benchmark", "value"),
            State("picker-train", "start_date"),
            State("picker-train", "end_date"),
            State("picker-test", "start_date"),
            State("picker-test", "end_date"),
            State("saved-ml-model-back", "data"),
            State("saved-ml-spec-back", "data"),
            State("saved-pick-num-back", "data"),
            State("saved-scen-model-back", "data"),
            State("saved-scen-spec-back", "data"),
            State("saved-benchmark-back", "data"),
            State("saved-perf-figure-page-2", "data"),
            State("saved-comp-figure-page-2", "data"),
            State("saved-universe-figure-page-2", "data"),
            State("select-solver", "value"),
            State("saved-solver", "data"),
            State("select-optimization-model", "value"),
            State("saved-optimization-model", "data"),
            State("slider-trading-sizes", "value"),
        ],
    )
    def plot_backtest(
        click,
        model,
        model_spec,
        pick_top,
        scen_model,
        scen_spec,
        benchmark,
        start_data,
        end_train,
        start_test,
        end_data,
        saved_model,
        saved_model_spec,
        saved_pick_top,
        saved_scen_model,
        saved_scen_spec,
        saved_benchmark,
        saved_perf_figure,
        saved_comp_figure,
        saved_universe_figure,
        solver,
        saved_solver,
        optimization_model,
        saved_optimization_model,
        lower_bound,
    ):
        # Initialize
        opt_init = ["Optimal", "Optimal Portfolio", "Optimal Portfolio", 3]
        bench_init = ["Benchmark", "Benchmark Portfolio", "Benchmark Portfolio", 3]

        if click:
            # RUN ML algo
            if model == "MST":
                _, subset_of_assets = algo.mst(
                    start_date=start_data, end_date=end_train, n_mst_runs=model_spec
                )
            else:
                _, subset_of_assets = algo.clustering(
                    start_date=start_data,
                    end_date=end_train,
                    n_clusters=model_spec,
                    n_assets=pick_top,
                )
            # RUN THE BACKTEST
            opt_table, bench_table, fig_performance, fig_composition = algo.backtest(
                start_train_date=start_data,
                start_test_date=start_test,
                end_test_date=end_data,
                subset_of_assets=subset_of_assets,
                benchmarks=benchmark,
                scenarios_type=scen_model,
                n_simulations=scen_spec,
                model=optimization_model,
                solver=solver,
                lower_bound=lower_bound,
            )
            # Save page values
            perf_figure = dcc.Graph(
                figure=fig_performance, style={"margin": "0%", "height": "800px"}
            )
            comp_figure = dcc.Graph(figure=fig_composition, style={"margin": "0%"})

            fig_universe = algo.plot_dots(
                start_date=start_test,
                end_date=end_data,
                optimal_portfolio=opt_table.iloc[0].to_list() + opt_init,
                benchmark=bench_table.iloc[0].to_list() + bench_init,
            )

            generated_figure = dcc.Graph(
                figure=fig_universe, style={"margin": "0%", "height": "1200px"}
            )

            return (
                perf_figure,
                comp_figure,
                model,
                model_spec,
                pick_top,
                scen_model,
                scen_spec,
                benchmark,
                model,
                model_spec,
                pick_top,
                scen_model,
                scen_spec,
                benchmark,
                perf_figure,
                comp_figure,
                True,
                generated_figure,
                generated_figure,
                solver,
                solver,
                optimization_model,
                optimization_model,
            )
        else:
            return (
                saved_perf_figure,
                saved_comp_figure,
                saved_model,
                saved_model_spec,
                saved_pick_top,
                saved_scen_model,
                saved_scen_spec,
                saved_benchmark,
                saved_model,
                saved_model_spec,
                saved_pick_top,
                saved_scen_model,
                saved_scen_spec,
                saved_benchmark,
                saved_perf_figure,
                saved_comp_figure,
                True,
                saved_universe_figure,
                saved_universe_figure,
                saved_solver,
                saved_solver,
                saved_optimization_model,
                saved_optimization_model,
            )

    @app.callback(
        Output("slider-output-container2", "children"), Input("my-slider2", "value")
    )
    def update_output(value):
        return "# of scenarios: {}".format(value)

    @app.callback(
        Output("slider-output-container-2-lifecycle", "children"),
        Input("my-slider-2-lifecycle", "value"),
    )
    def update_output_lifecycle(value):
        return "# of scenarios: {}".format(value)

    @app.callback(
        Output(
            component_id="slider-trading-sizes-container", component_property="style"
        ),
        [Input(component_id="select-solver", component_property="value")],
    )
    def show_hide_element(solver):
        if solver == "ECOS_BB" or solver == "CPLEX" or solver == "MOSEK":
            return {"display": "block"}
        else:
            return {"display": "none"}

    @app.callback(
        Output("slider-output-container-backtest", "children"),
        [Input("slider-backtest", "value")],
    )
    def update_output_cluster(value):
        return "In case of CLUSTERING: # of the best performing assets selected from each cluster: {}".format(
            value
        )

    @app.callback(
        Output("slider-output-container-backtest-ml", "children"),
        [Input("slider-backtest-ml", "value")],
    )
    def update_output_ml_type(value):
        return "# of clusters or # of MST runs: {}".format(value)

    @app.callback(
        Output(
            component_id="slider-output-container-lifecycle", component_property="style"
        ),
        Input(component_id="select-ml-lifecycle", component_property="value"),
    )
    def update_output_cluster_lifecycle(value):
        if value == "Clustering":
            return {"display": "block"}
        else:
            return {"display": "none"}

    @app.callback(
        Output(component_id="if-top-performers", component_property="style"),
        Input(component_id="top-performers", component_property="value"),
    )
    def update_output_top_performers(value):
        if value == "yes":
            return {"display": "block"}
        else:
            return {"display": "none"}

    @app.callback(
        Output("slider-output-container-lifecycle-ml", "children"),
        [Input("slider-lifecycle-ml", "value")],
    )
    def update_output_ml_type_lifecycle(value):
        return "# of clusters or # of MST runs: {}".format(value)

    @app.callback(
        Output("slider-trading-sizes-output", "children"),
        [Input("slider-trading-sizes", "value")],
    )
    def update_trading_sizes(value):
        return "Minimum required asset weight in the portfolio: {}%".format(value)

    @app.callback(
        [
            Output("picker-test", "start_date"),
            Output("picker-test", "end_date"),
            Output("picker-train", "start_date"),
            Output("picker-train", "end_date"),
            Output("saved-split-date", "data"),
        ],
        Input("picker-train", "end_date"),
        State("saved-split-date", "data"),
    )
    def update_test_date(selected_date, saved_split_date):
        if selected_date:
            split_date = selected_date
        else:
            # TODO change this hardcoded date
            split_date = saved_split_date

        return split_date, algo.max_date, algo.min_date, split_date, split_date

    # AI Feature Selection
    # -----------------------------------------------------------------------------------------------------------------
    # PLOT ML MST GRAPH
    @app.callback(
        [
            Output("mlFig", "children"),
            Output("picker-AI", "start_date"),
            Output("picker-AI", "end_date"),
            Output("AIResult", "data"),
            Output("AInumber", "children"),
            Output("model-dropdown", "value"),
            Output("ML-num-dropdown", "value"),
            Output("saved-start-date-page-1", "data"),
            Output("saved-end-date-page-1", "data"),
            Output("saved-ai-table", "data"),
            Output("saved-ml-model", "data"),
            Output("saved-ml-spec", "data"),
            Output("saved-ml-text", "data"),
            Output("saved-figure-page-1", "data"),
            Output("loading-output-ml", "children"),
        ],
        [Input("MLRun", "n_clicks")],
        [
            State("model-dropdown", "value"),
            State("ML-num-dropdown", "value"),
            State("picker-AI", "start_date"),
            State("picker-AI", "end_date"),
            State("saved-start-date-page-1", "data"),
            State("saved-end-date-page-1", "data"),
            State("saved-ai-table", "data"),
            State("saved-ml-model", "data"),
            State("saved-ml-spec", "data"),
            State("saved-ml-text", "data"),
            State("saved-figure-page-1", "data"),
        ],
    )
    def plot_ml(
        click_ml,
        model,
        spec,
        start,
        end,
        saved_start,
        saved_end,
        saved_ai_table,
        saved_model,
        saved_spec,
        saved_text,
        saved_figure,
    ):
        if click_ml:
            selected_start = str(start)
            selected_end = str(end)

            # MST
            if model == "MST":
                # RUN THE MINIMUM SPANNING TREE METHOD
                fig, ai_subset = algo.mst(
                    start_date=selected_start,
                    end_date=selected_end,
                    n_mst_runs=spec,
                    plot=True,
                )
                generated_figure = dcc.Graph(
                    figure=fig, style={"height": "800px", "margin": "0%"}
                )

            # CLUSTERING
            else:
                fig, ai_subset = algo.clustering(
                    start_date=selected_start,
                    end_date=selected_end,
                    n_clusters=spec,
                    n_assets=10,
                    plot=True,
                )
                generated_figure = dcc.Graph(
                    figure=fig, style={"height": "800px", "margin": "0%"}
                )

            ai_data = algo.get_stat(start_date=selected_start, end_date=selected_end)
            ai_table = ai_data.loc[
                list(ai_subset),
                [
                    "Name",
                    "ISIN",
                    "Sharpe Ratio",
                    "Average Annual Returns",
                    "Standard Deviation of Returns",
                ],
            ]
            # ROUNDING
            ai_table["Standard Deviation of Returns"] = round(
                ai_table["Standard Deviation of Returns"], 2
            )
            ai_table["Average Annual Returns"] = round(
                ai_table["Average Annual Returns"], 2
            )

            ml_text = "Number of selected assets: " + str(len(ai_table))

            return (
                generated_figure,
                selected_start,
                selected_end,
                ai_table.to_dict("records"),
                ml_text,
                model,
                spec,
                selected_start,
                selected_end,
                ai_table.to_dict("records"),
                model,
                spec,
                ml_text,
                generated_figure,
                True,
            )
        else:
            return (
                saved_figure,
                saved_start,
                saved_end,
                saved_ai_table,
                saved_text,
                saved_model,
                saved_spec,
                saved_start,
                saved_end,
                saved_ai_table,
                saved_model,
                saved_spec,
                saved_text,
                saved_figure,
                True,
            )

    # MARKET OVERVIEW
    # -----------------------------------------------------------------------------------------------------------------
    # PLOT GRAPH WITH DOTS
    @app.callback(
        [
            Output("dotsFig", "children"),
            Output("picker-show", "start_date"),
            Output("picker-show", "end_date"),
            Output("find-fund", "value"),
            Output("saved-start-date-page-0", "data"),
            Output("saved-end-date-page-0", "data"),
            Output("saved-find-fund", "data"),
            Output("saved-figure-page-0", "data"),
            Output("loading-output-dots", "children"),
            Output("saved-top-performers-names", "data"),
            Output("top-performers", "value"),
            Output("combine-top-performers", "value"),
            Output("top-performers-pct", "value"),
            Output("saved-top-performers", "data"),
            Output("saved-combine-top-performers", "data"),
            Output("saved-top-performers-pct", "data"),
        ],
        [Input("page-content", "children"), Input("show", "n_clicks")],
        [
            State("picker-show", "start_date"),
            State("picker-show", "end_date"),
            State("find-fund", "value"),
            State("saved-start-date-page-0", "data"),
            State("saved-end-date-page-0", "data"),
            State("saved-find-fund", "data"),
            State("saved-figure-page-0", "data"),
            State("top-performers", "value"),
            State("combine-top-performers", "value"),
            State("top-performers-pct", "value"),
            State("saved-top-performers-names", "data"),
            State("saved-top-performers", "data"),
            State("saved-combine-top-performers", "data"),
            State("saved-top-performers-pct", "data"),
        ],
    )
    def plot_dots(
        trigger,
        click,
        start,
        end,
        search,
        saved_start,
        saved_end,
        saved_find_fund,
        saved_figure,
        top_performers,
        combine_top_performers,
        top_performers_pct,
        saved_top_performers_names,
        saved_top_performers,
        saved_combine_top_performers,
        saved_top_performers_pct,
    ):
        if click:
            selected_start = str(start)
            selected_end = str(end)

            if top_performers == "yes":
                top_assets = algo.get_top_performing_assets(
                    time_periods=[(selected_start, selected_end)],
                    top_percent=top_performers_pct / 100,
                )
                if combine_top_performers == "yes":
                    top_assets = [
                        asset
                        for asset in top_assets
                        if asset in saved_top_performers_names
                    ]
                saved_top_performers_names = top_assets
            else:
                top_assets = []
                saved_top_performers_names = []

            fig = algo.plot_dots(
                start_date=selected_start,
                end_date=selected_end,
                fund_set=search,
                top_performers=top_assets,
            )
            generated_figure = dcc.Graph(
                figure=fig,
                style={
                    "position": "absolute",
                    "right": "0%",
                    "bottom": "0%",
                    "top": "0%",
                    "left": "0%",
                },
            )
            return (
                generated_figure,
                selected_start,
                selected_end,
                search,
                selected_start,
                selected_end,
                search,
                generated_figure,
                True,
                saved_top_performers_names,
                top_performers,
                combine_top_performers,
                top_performers_pct,
                top_performers,
                combine_top_performers,
                top_performers_pct,
            )
        else:
            return (
                saved_figure,
                saved_start,
                saved_end,
                saved_find_fund,
                saved_start,
                saved_end,
                saved_find_fund,
                saved_figure,
                True,
                saved_top_performers_names,
                saved_top_performers,
                saved_combine_top_performers,
                saved_top_performers_pct,
                saved_top_performers,
                saved_combine_top_performers,
                saved_top_performers_pct,
            )
