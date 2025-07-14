from typing import Any

from dash import Output, State, dcc
from pydantic import BaseModel


class BacktestInputs(BaseModel):
    model: str
    model_spec: int
    pick_top: int
    scen_model: str
    scen_spec: int
    benchmark: list[str]
    start_train_date: str
    end_train_date: str
    start_test_date: str
    end_test_date: str
    solver: str
    optimization_model: str
    lower_bound: int

    @classmethod
    def as_state_vector(cls) -> list[State]:
        return [
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
            State("select-solver", "value"),
            State("select-optimization-model", "value"),
            State("slider-trading-sizes", "value"),
        ]

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "BacktestInputs":
        return cls(**kwargs)

# --- Outputs Model ---
class BacktestOutputs(BaseModel):
    perf_figure: Any
    comp_figure: Any
    universe_figure: Any

    @classmethod
    def as_output_vector(cls) -> list[Output]:
        return [
            Output("backtestPerfFig", "children"),
            Output("backtestCompFig", "children"),
            Output("backtestUniverseFig", "children"),
        ]

    def as_tuple(self) -> tuple[Any]:
        return tuple(self.__dict__.values())

def plot_backtest(algo, inputs):
    # Initialize
    opt_init = ["Optimal", 3, 3, 3] #"Optimal Portfolio", "Optimal Portfolio", 3]
    bench_init = ["Benchmark", 4,4,4] #"Benchmark Portfolio", "Benchmark Portfolio", 3]

    # RUN ML algo
    if inputs.model == "MST":
        _, subset_of_assets = algo.mst(
            start_date=inputs.start_train_date,
            end_date=inputs.end_train_date,
            n_mst_runs=inputs.model_spec
        )
    else:
        _, subset_of_assets = algo.clustering(
            start_date=inputs.start_train_date,
            end_date=inputs.end_train_date,
            n_clusters=inputs.model_spec,
            n_assets=inputs.pick_top,
        )

    # RUN THE BACKTEST
    opt_table, bench_table, fig_performance, fig_composition = algo.backtest(
        start_train_date=inputs.start_train_date,
        start_test_date=inputs.start_test_date,
        end_test_date=inputs.end_test_date,
        subset_of_assets=subset_of_assets,
        benchmarks=inputs.benchmark,
        scenarios_type=inputs.scen_model,
        n_simulations=inputs.scen_spec,
        model=inputs.optimization_model,
        solver=inputs.solver,
        lower_bound=inputs.lower_bound,
    )

    print(opt_table)
    print(bench_table)
    print(opt_table.iloc[0].to_list() + opt_init)
    print(bench_table.iloc[0].to_list() + bench_init)
    #assert False

    # Create figures
    perf_figure = dcc.Graph(
        figure=fig_performance, style={"margin": "0%", "height": "800px"}
    )

    comp_figure = dcc.Graph(
        figure=fig_composition, style={"margin": "0%"}
    )

    fig_universe = algo.plot_dots(
        start_date=inputs.start_test_date,
        end_date=inputs.end_test_date,
        optimal_portfolio=opt_table.iloc[0].to_list() + opt_init,
        benchmark=bench_table.iloc[0].to_list() + bench_init,
    )

    # try:
    # fig_universe.show()

    universe_figure = dcc.Graph(
        figure=fig_universe, style={"margin": "0%", "height": "1200px"}
    )

    # Create outputs
    outputs = BacktestOutputs(
        perf_figure=perf_figure,
        comp_figure=comp_figure,
        universe_figure=universe_figure
    )

    return outputs
