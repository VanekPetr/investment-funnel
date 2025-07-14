"""
Backtesting model module for the Investment Funnel dashboard.

This module provides the data models and processing logic for the Backtest page.
It includes functionality for testing investment strategies using historical data,
with support for different optimization models, machine learning methods, and
scenario generation techniques.
"""

from typing import Any

from dash import Output, State, dcc
from pydantic import BaseModel


class BacktestInputs(BaseModel):
    """
    Input model for Backtesting functionality.

    This Pydantic model defines the input parameters required for the backtesting
    process, including model selection, date ranges, and optimization parameters.

    Attributes:
        model (str): The feature selection model type (e.g., "MST" or "Clustering")
        model_spec (int): The number of clusters or MST runs to perform
        pick_top (int): Number of top-performing assets to select from each cluster
        scen_model (str): The scenario generation model (e.g., "Bootstrap" or "MonteCarlo")
        scen_spec (int): The number of scenarios to generate
        benchmark (list[str]): List of benchmark assets to compare against
        start_train_date (str): Start date for the training period
        end_train_date (str): End date for the training period
        start_test_date (str): Start date for the testing period
        end_test_date (str): End date for the testing period
        solver (str): The optimization solver to use
        optimization_model (str): The portfolio optimization model to use
        lower_bound (int): Lower bound for asset allocation proportions
    """
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
        """
        Create a list of Dash State objects for callback input capturing.

        This method maps the model's attributes to the corresponding UI components
        to capture their values in a Dash callback.

        Returns:
            list[State]: A list of Dash State objects for the callback
        """
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
        """
        Create a BacktestInputs instance from keyword arguments.

        This factory method creates a new instance of the class from
        the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments matching the model's fields

        Returns:
            BacktestInputs: A new instance of the BacktestInputs class
        """
        return cls(**kwargs)

# --- Outputs Model ---
class BacktestOutputs(BaseModel):
    """
    Output model for Backtesting functionality.

    This Pydantic model defines the output data structure for the backtesting
    process, including performance figures, composition analysis, and universe visualization.

    Attributes:
        perf_figure (Any): The performance comparison figure (Dash component)
        comp_figure (Any): The portfolio composition figure (Dash component)
        universe_figure (Any): The investment universe visualization (Dash component)
    """
    perf_figure: Any
    comp_figure: Any
    universe_figure: Any

    @classmethod
    def as_output_vector(cls) -> list[Output]:
        """
        Create a list of Dash Output objects for callback output mapping.

        This method maps the model's attributes to the corresponding UI components
        that will display the results in the Dash application.

        Returns:
            list[Output]: A list of Dash Output objects for the callback
        """
        return [
            Output("backtestPerfFig", "children"),
            Output("backtestCompFig", "children"),
            Output("backtestUniverseFig", "children"),
        ]

    def as_tuple(self) -> tuple[Any]:
        """
        Convert the model instance to a tuple of values.

        This method is used to prepare the model's data for returning
        from a Dash callback function.

        Returns:
            tuple[Any]: A tuple containing all the model's attribute values
        """
        return tuple(self.__dict__.values())

def plot_backtest(algo, inputs):
    """
    Perform backtesting and generate visualization outputs.

    This function processes the input parameters to run a complete backtesting workflow:
    1. Selects assets using the specified feature selection method (MST or Clustering)
    2. Runs the backtest with the selected optimization model and solver
    3. Generates performance comparison, portfolio composition, and universe visualization figures

    Args:
        algo: The algorithm object containing the backtesting methods
        inputs (BacktestInputs): The input parameters for the backtesting process

    Returns:
        BacktestOutputs: The output data containing the visualization figures
    """
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
