"""
Tests for the model functions in the pages/models directory.

This module contains tests for the model functions that power the dashboard pages.
Each test verifies that a specific model function correctly processes input data
and generates the expected output visualizations.
"""

from typing import Any
from dash import Output, State

from funnel.pages.models.ai_feature import FeatureInput, FeatureOutput, plot_ml
from funnel.pages.models.backtest import BacktestInputs, BacktestOutputs, plot_backtest
from funnel.pages.models.lifecycle import LifecycleInputs, LifecycleOutputs, plot_lifecycle
from funnel.pages.models.overview import OverviewInputs, OverviewOutputs, plot_overview


def test_feature_input_from_kwargs() -> None:
    """
    Test that FeatureInput.from_kwargs creates a valid instance.

    This test verifies that the from_kwargs class method correctly
    creates a new instance of the FeatureInput class from keyword arguments.
    """
    # Create a FeatureInput instance using from_kwargs
    kwargs = {
        "model": "MST",
        "spec": 4,
        "start_date": "2013-01-01",
        "end_date": "2014-01-01"
    }

    feature_input = FeatureInput.from_kwargs(**kwargs)

    # Verify that the instance has the correct attributes
    assert feature_input.model == "MST"
    assert feature_input.spec == 4
    assert feature_input.start_date == "2013-01-01"
    assert feature_input.end_date == "2014-01-01"


def test_feature_output_as_tuple() -> None:
    """
    Test that FeatureOutput.as_tuple returns the correct tuple.

    This test verifies that the as_tuple method correctly converts
    the FeatureOutput instance to a tuple of its attribute values.
    """
    # Create a FeatureOutput instance
    feature_output = FeatureOutput(
        ml_figure="test_figure",
        ai_result="test_result",
        ai_number="test_number"
    )

    # Call the method being tested
    result = feature_output.as_tuple()

    # Verify that the result is a tuple with the correct values
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[0] == "test_figure"
    assert result[1] == "test_result"
    assert result[2] == "test_number"


def test_feature_input_as_state_vector() -> None:
    """
    Test that FeatureInput.as_state_vector returns the correct list of State objects.

    This test verifies that the as_state_vector class method correctly
    returns a list of Dash State objects for callback input capturing.
    """
    # Call the method being tested
    result = FeatureInput.as_state_vector()

    # Verify that the result is a list of State objects with the correct IDs and properties
    assert isinstance(result, list)
    assert len(result) == 4
    assert all(isinstance(state, State) for state in result)
    assert result[0].component_id == "model-dropdown"
    assert result[0].component_property == "value"
    assert result[1].component_id == "ML-num-dropdown"
    assert result[1].component_property == "value"
    assert result[2].component_id == "picker-AI"
    assert result[2].component_property == "start_date"
    assert result[3].component_id == "picker-AI"
    assert result[3].component_property == "end_date"


def test_feature_output_as_output_vector() -> None:
    """
    Test that FeatureOutput.as_output_vector returns the correct list of Output objects.

    This test verifies that the as_output_vector class method correctly
    returns a list of Dash Output objects for callback output mapping.
    """
    # Call the method being tested
    result = FeatureOutput.as_output_vector()

    # Verify that the result is a list of Output objects with the correct IDs and properties
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(output, Output) for output in result)
    assert result[0].component_id == "mlFig"
    assert result[0].component_property == "children"
    assert result[1].component_id == "AIResult"
    assert result[1].component_property == "data"
    assert result[2].component_id == "AInumber"
    assert result[2].component_property == "children"


def test_lifecycle_input_from_kwargs() -> None:
    """
    Test that LifecycleInputs.from_kwargs creates a valid instance.

    This test verifies that the from_kwargs class method correctly
    creates a new instance of the LifecycleInputs class from keyword arguments.
    """
    # Create a LifecycleInputs instance using from_kwargs
    kwargs = {
        "model": "MST",
        "model_spec": 2,
        "pick_top": 3,
        "scen_model": "Bootstrap",
        "scen_spec": 200,
        "start_date": "2019-01-01",
        "end_date": "2020-01-01",
        "end_year": 2030,
        "portfolio_value": 100000,
        "yearly_withdraws": 1000,
        "risk_preference": 15
    }

    lifecycle_input = LifecycleInputs.from_kwargs(**kwargs)

    # Verify that the instance has the correct attributes
    assert lifecycle_input.model == "MST"
    assert lifecycle_input.model_spec == 2
    assert lifecycle_input.pick_top == 3
    assert lifecycle_input.scen_model == "Bootstrap"
    assert lifecycle_input.scen_spec == 200
    assert lifecycle_input.start_date == "2019-01-01"
    assert lifecycle_input.end_date == "2020-01-01"
    assert lifecycle_input.end_year == 2030
    assert lifecycle_input.portfolio_value == 100000
    assert lifecycle_input.yearly_withdraws == 1000
    assert lifecycle_input.risk_preference == 15


def test_lifecycle_output_as_tuple() -> None:
    """
    Test that LifecycleOutputs.as_tuple returns the correct tuple.

    This test verifies that the as_tuple method correctly converts
    the LifecycleOutputs instance to a tuple of its attribute values.
    """
    # Create a LifecycleOutputs instance
    lifecycle_output = LifecycleOutputs(
        glidepaths_output_fig="test_glidepaths_figure",
        performance_output_fig="test_performance_figure",
        lifecycle_all_output_fig="test_lifecycle_all_figure"
    )

    # Call the method being tested
    result = lifecycle_output.as_tuple()

    # Verify that the result is a tuple with the correct values
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[0] == "test_glidepaths_figure"
    assert result[1] == "test_performance_figure"
    assert result[2] == "test_lifecycle_all_figure"


def test_lifecycle_input_as_state_vector() -> None:
    """
    Test that LifecycleInputs.as_state_vector returns the correct list of State objects.

    This test verifies that the as_state_vector class method correctly
    returns a list of Dash State objects for callback input capturing.
    """
    # Call the method being tested
    result = LifecycleInputs.as_state_vector()

    # Verify that the result is a list of State objects with the correct IDs and properties
    assert isinstance(result, list)
    assert len(result) == 11
    assert all(isinstance(state, State) for state in result)
    # Check a few key states
    assert result[0].component_id == "select-ml-lifecycle"
    assert result[0].component_property == "value"
    assert result[1].component_id == "slider-lifecycle-ml"
    assert result[1].component_property == "value"
    assert result[2].component_id == "slider-lifecycle"
    assert result[2].component_property == "value"
    assert result[5].component_id == "picker-lifecycle"
    assert result[5].component_property == "start_date"
    assert result[6].component_id == "picker-lifecycle"
    assert result[6].component_property == "end_date"


def test_lifecycle_output_as_output_vector() -> None:
    """
    Test that LifecycleOutputs.as_output_vector returns the correct list of Output objects.

    This test verifies that the as_output_vector class method correctly
    returns a list of Dash Output objects for callback output mapping.
    """
    # Call the method being tested
    result = LifecycleOutputs.as_output_vector()

    # Verify that the result is a list of Output objects with the correct IDs and properties
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(output, Output) for output in result)
    assert result[0].component_id == "glidepaths-output-fig"
    assert result[0].component_property == "children"
    assert result[1].component_id == "performance-output-fig"
    assert result[1].component_property == "children"
    assert result[2].component_id == "lifecycle-all-output-fig"
    assert result[2].component_property == "children"


def test_plot_lifecycle_with_mst(algo: Any) -> None:
    """
    Test that plot_lifecycle generates new figures when clicked with model="MST".

    This test verifies that the plot_lifecycle function correctly processes
    input parameters and generates the expected visualization outputs for
    lifecycle investment analysis when using the MST model.

    Args:
        algo (Any): The investment bot fixture from conftest.py
    """
    # Create test input parameters
    inputs = LifecycleInputs(
        model="MST",  # Feature selection model type
        model_spec=2,  # Number of MST runs
        pick_top=3,  # Number of top assets to select
        scen_model="Bootstrap",  # Scenario generation method
        scen_spec=200,  # Number of scenarios to generate
        start_date="2019-01-01",  # Analysis start date
        end_date="2020-01-01",  # Analysis end date
        end_year=2030,  # Target end year
        portfolio_value=100000,  # Initial portfolio value
        yearly_withdraws=1000,  # Annual withdrawal amount
        risk_preference=15,  # Initial risk appetite percentage
    )

    # Call the function being tested
    result = plot_lifecycle(
        algo,  # Algorithm object with investment methods
        inputs  # Input parameters
    )

    # Verify the result type
    assert isinstance(result, LifecycleOutputs), "Should return a LifecycleOutputs object"

    # Verify the output figures were generated
    assert result.glidepaths_output_fig is not None, "Should return a glidepaths figure"
    assert result.performance_output_fig is not None, "Should return a performance figure"
    assert result.lifecycle_all_output_fig is not None, "Should return a lifecycle all figure"


def test_plot_lifecycle_with_clustering(algo: Any) -> None:
    """
    Test that plot_lifecycle generates new figures when clicked with model="Cluster".

    This test verifies that the plot_lifecycle function correctly processes
    input parameters and generates the expected visualization outputs for
    lifecycle investment analysis when using the Clustering model.

    Args:
        algo (Any): The investment bot fixture from conftest.py
    """
    # Create test input parameters
    inputs = LifecycleInputs(
        model="Cluster",  # Feature selection model type
        model_spec=3,  # Number of clusters
        pick_top=3,  # Number of top assets to select
        scen_model="Bootstrap",  # Scenario generation method
        scen_spec=200,  # Number of scenarios to generate
        start_date="2019-01-01",  # Analysis start date
        end_date="2020-01-01",  # Analysis end date
        end_year=2030,  # Target end year
        portfolio_value=100000,  # Initial portfolio value
        yearly_withdraws=1000,  # Annual withdrawal amount
        risk_preference=15,  # Initial risk appetite percentage
    )

    # Call the function being tested
    result = plot_lifecycle(
        algo,  # Algorithm object with investment methods
        inputs  # Input parameters
    )

    # Verify the result type
    assert isinstance(result, LifecycleOutputs), "Should return a LifecycleOutputs object"

    # Verify the output figures were generated
    assert result.glidepaths_output_fig is not None, "Should return a glidepaths figure"
    assert result.performance_output_fig is not None, "Should return a performance figure"
    assert result.lifecycle_all_output_fig is not None, "Should return a lifecycle all figure"


def test_backtest_input_from_kwargs() -> None:
    """
    Test that BacktestInputs.from_kwargs creates a valid instance.

    This test verifies that the from_kwargs class method correctly
    creates a new instance of the BacktestInputs class from keyword arguments.
    """
    # Create a BacktestInputs instance using from_kwargs
    kwargs = {
        "start_train_date": "2018-01-01",
        "end_train_date": "2021-12-31",
        "start_test_date": "2022-01-01",
        "end_test_date": "2023-01-01",
        "model": "MST",
        "model_spec": 2,
        "pick_top": 5,
        "scen_model": "Bootstrap",
        "scen_spec": 1000,
        "benchmark": ["Wealth Invest Amalie Global AK"],
        "solver": "CLARABEL",
        "optimization_model": "CVaR model",
        "lower_bound": 0
    }

    backtest_input = BacktestInputs.from_kwargs(**kwargs)

    # Verify that the instance has the correct attributes
    assert backtest_input.start_train_date == "2018-01-01"
    assert backtest_input.end_train_date == "2021-12-31"
    assert backtest_input.start_test_date == "2022-01-01"
    assert backtest_input.end_test_date == "2023-01-01"
    assert backtest_input.model == "MST"
    assert backtest_input.model_spec == 2
    assert backtest_input.pick_top == 5
    assert backtest_input.scen_model == "Bootstrap"
    assert backtest_input.scen_spec == 1000
    assert backtest_input.benchmark == ["Wealth Invest Amalie Global AK"]
    assert backtest_input.solver == "CLARABEL"
    assert backtest_input.optimization_model == "CVaR model"
    assert backtest_input.lower_bound == 0


def test_backtest_output_as_tuple() -> None:
    """
    Test that BacktestOutputs.as_tuple returns the correct tuple.

    This test verifies that the as_tuple method correctly converts
    the BacktestOutputs instance to a tuple of its attribute values.
    """
    # Create a BacktestOutputs instance
    backtest_output = BacktestOutputs(
        perf_figure="test_perf_figure",
        comp_figure="test_comp_figure",
        universe_figure="test_universe_figure"
    )

    # Call the method being tested
    result = backtest_output.as_tuple()

    # Verify that the result is a tuple with the correct values
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[0] == "test_perf_figure"
    assert result[1] == "test_comp_figure"
    assert result[2] == "test_universe_figure"


def test_backtest_input_as_state_vector() -> None:
    """
    Test that BacktestInputs.as_state_vector returns the correct list of State objects.

    This test verifies that the as_state_vector class method correctly
    returns a list of Dash State objects for callback input capturing.
    """
    # Call the method being tested
    result = BacktestInputs.as_state_vector()

    # Verify that the result is a list of State objects with the correct IDs and properties
    assert isinstance(result, list)
    assert len(result) == 13
    assert all(isinstance(state, State) for state in result)
    # Check a few key states
    assert result[0].component_id == "select-ml"
    assert result[0].component_property == "value"
    assert result[1].component_id == "slider-backtest-ml"
    assert result[1].component_property == "value"
    assert result[2].component_id == "slider-backtest"
    assert result[2].component_property == "value"
    assert result[6].component_id == "picker-train"
    assert result[6].component_property == "start_date"
    assert result[7].component_id == "picker-train"
    assert result[7].component_property == "end_date"
    assert result[8].component_id == "picker-test"
    assert result[8].component_property == "start_date"


def test_backtest_output_as_output_vector() -> None:
    """
    Test that BacktestOutputs.as_output_vector returns the correct list of Output objects.

    This test verifies that the as_output_vector class method correctly
    returns a list of Dash Output objects for callback output mapping.
    """
    # Call the method being tested
    result = BacktestOutputs.as_output_vector()

    # Verify that the result is a list of Output objects with the correct IDs and properties
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(output, Output) for output in result)
    assert result[0].component_id == "backtestPerfFig"
    assert result[0].component_property == "children"
    assert result[1].component_id == "backtestCompFig"
    assert result[1].component_property == "children"
    assert result[2].component_id == "backtestUniverseFig"
    assert result[2].component_property == "children"


def test_plot_backtest_with_mst(algo: Any) -> None:
    """
    Test that plot_backtest generates new figures when clicked with model="MST".

    This test verifies that the plot_backtest function correctly processes
    input parameters and generates the expected visualization outputs for
    backtesting investment strategies when using the MST model.

    Args:
        algo (Any): The investment bot fixture from conftest.py
    """
    # Create test input parameters
    inputs = BacktestInputs(
        start_train_date="2018-01-01",  # Training period start date
        end_train_date="2021-12-31",    # Training period end date
        start_test_date="2022-01-01",   # Testing period start date
        end_test_date="2023-01-01",     # Testing period end date
        model="MST",                    # Feature selection model type
        model_spec=2,                   # Number of MST runs
        pick_top=5,                     # Number of top assets to select
        scen_model="Bootstrap",         # Scenario generation method
        scen_spec=1000,                 # Number of scenarios to generate
        benchmark=["Wealth Invest Amalie Global AK"],  # Benchmark assets
        solver="CLARABEL",              # Optimization solver
        optimization_model="CVaR model",  # Portfolio optimization model
        lower_bound=0                   # Lower bound for asset allocation
    )

    # Call the function being tested
    result = plot_backtest(
        algo,  # Algorithm object with investment methods
        inputs  # Input parameters
    )

    # Verify the result type
    assert isinstance(result, BacktestOutputs), "Should return a BacktestOutputs object"


def test_plot_backtest_with_clustering(algo: Any) -> None:
    """
    Test that plot_backtest generates new figures when clicked with model="Cluster".

    This test verifies that the plot_backtest function correctly processes
    input parameters and generates the expected visualization outputs for
    backtesting investment strategies when using the Clustering model.

    Args:
        algo (Any): The investment bot fixture from conftest.py
    """
    # Create test input parameters
    inputs = BacktestInputs(
        start_train_date="2018-01-01",  # Training period start date
        end_train_date="2021-12-31",    # Training period end date
        start_test_date="2022-01-01",   # Testing period start date
        end_test_date="2023-01-01",     # Testing period end date
        model="Cluster",                # Feature selection model type
        model_spec=3,                   # Number of clusters
        pick_top=5,                     # Number of top assets to select
        scen_model="Bootstrap",         # Scenario generation method
        scen_spec=1000,                 # Number of scenarios to generate
        benchmark=["Wealth Invest Amalie Global AK"],  # Benchmark assets
        solver="CLARABEL",              # Optimization solver
        optimization_model="CVaR model",  # Portfolio optimization model
        lower_bound=0                   # Lower bound for asset allocation
    )

    # Call the function being tested
    result = plot_backtest(
        algo,  # Algorithm object with investment methods
        inputs  # Input parameters
    )

    # Verify the result type
    assert isinstance(result, BacktestOutputs), "Should return a BacktestOutputs object"


def test_plot_ml(algo: Any) -> None:
    """
    Test that plot_ml generates new figures when clicked.

    This test verifies that the plot_ml function correctly processes
    input parameters and generates the expected visualization outputs for
    AI feature selection.

    Args:
        algo (Any): The investment bot fixture from conftest.py
    """
    # Create test input parameters
    inputs = FeatureInput(
        start_date="2013-01-01",  # Analysis start date
        end_date="2014-01-01",    # Analysis end date
        model="MST",              # AI model type (Minimum Spanning Tree)
        spec=4                    # Number of MST runs
    )

    # Call the function being tested
    result = plot_ml(
        algo,  # Algorithm object with investment methods
        inputs  # Input parameters
    )

    # Verify the result type and output figure
    assert isinstance(result, FeatureOutput), "Should return a FeatureOutput object"
    assert result.ml_figure is not None, "Should return a ml figure"


def test_overview_input_from_kwargs() -> None:
    """
    Test that OverviewInputs.from_kwargs creates a valid instance.

    This test verifies that the from_kwargs class method correctly
    creates a new instance of the OverviewInputs class from keyword arguments.
    """
    # Create an OverviewInputs instance using from_kwargs
    kwargs = {
        "start_date": "2013-01-01",
        "end_date": "2023-01-01",
        "search": ["Wealth Invest Amalie Global AK"],
        "top_performers": "yes"
    }

    overview_input = OverviewInputs.from_kwargs(**kwargs)

    # Verify that the instance has the correct attributes
    assert overview_input.start_date == "2013-01-01"
    assert overview_input.end_date == "2023-01-01"
    assert overview_input.search == ["Wealth Invest Amalie Global AK"]
    assert overview_input.top_performers == "yes"


def test_overview_output_as_tuple() -> None:
    """
    Test that OverviewOutputs.as_tuple returns the correct tuple.

    This test verifies that the as_tuple method correctly converts
    the OverviewOutputs instance to a tuple of its attribute values.
    """
    # Create an OverviewOutputs instance
    overview_output = OverviewOutputs(
        dots_figure="test_figure"
    )

    # Call the method being tested
    result = overview_output.as_tuple()

    # Verify that the result is a tuple with the correct values
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0] == "test_figure"


def test_overview_input_as_state_vector() -> None:
    """
    Test that OverviewInputs.as_state_vector returns the correct list of State objects.

    This test verifies that the as_state_vector class method correctly
    returns a list of Dash State objects for callback input capturing.
    """
    # Call the method being tested
    result = OverviewInputs.as_state_vector()

    # Verify that the result is a list of State objects with the correct IDs and properties
    assert isinstance(result, list)
    assert len(result) == 4
    assert all(isinstance(state, State) for state in result)
    assert result[0].component_id == "picker-show"
    assert result[0].component_property == "start_date"
    assert result[1].component_id == "picker-show"
    assert result[1].component_property == "end_date"
    assert result[2].component_id == "find-fund"
    assert result[2].component_property == "value"
    assert result[3].component_id == "top-performers"
    assert result[3].component_property == "value"


def test_overview_output_as_output_vector() -> None:
    """
    Test that OverviewOutputs.as_output_vector returns the correct list of Output objects.

    This test verifies that the as_output_vector class method correctly
    returns a list of Dash Output objects for callback output mapping.
    """
    # Call the method being tested
    result = OverviewOutputs.as_output_vector()

    # Verify that the result is a list of Output objects with the correct IDs and properties
    assert isinstance(result, list)
    assert len(result) == 1
    assert all(isinstance(output, Output) for output in result)
    assert result[0].component_id == "dotsFig"
    assert result[0].component_property == "children"


def test_plot_overview_with_top_performers(algo: Any) -> None:
    """
    Test that plot_overview generates new figures when clicked with top_performers="yes".

    This test verifies that the plot_overview function correctly processes
    input parameters and generates the expected visualization outputs for
    the market overview when top_performers is set to "yes".

    Args:
        algo (Any): The investment bot fixture from conftest.py
    """
    # Create test input parameters
    inputs = OverviewInputs(
        start_date="2013-01-01",  # Analysis start date
        end_date="2023-01-01",    # Analysis end date
        search=["Wealth Invest Amalie Global AK", "BankInvest Danske Aktier A"],  # Funds to highlight
        top_performers="yes"      # Whether to highlight top performers
    )

    # Call the function being tested
    result = plot_overview(
        algo,  # Algorithm object with investment methods
        inputs  # Input parameters
    )

    # Verify the result type and output figure
    assert isinstance(result, OverviewOutputs), "Should return an OverviewOutputs object"
    assert result.dots_figure is not None, "Should return a scatter plot figure"


def test_plot_overview_without_top_performers(algo: Any) -> None:
    """
    Test that plot_overview generates new figures when clicked with top_performers="no".

    This test verifies that the plot_overview function correctly processes
    input parameters and generates the expected visualization outputs for
    the market overview when top_performers is set to "no".

    Args:
        algo (Any): The investment bot fixture from conftest.py
    """
    # Create test input parameters
    inputs = OverviewInputs(
        start_date="2013-01-01",  # Analysis start date
        end_date="2023-01-01",    # Analysis end date
        search=["Wealth Invest Amalie Global AK", "BankInvest Danske Aktier A"],  # Funds to highlight
        top_performers="no"       # Whether to highlight top performers
    )

    # Call the function being tested
    result = plot_overview(
        algo,  # Algorithm object with investment methods
        inputs  # Input parameters
    )

    # Verify the result type and output figure
    assert isinstance(result, OverviewOutputs), "Should return an OverviewOutputs object"
    assert result.dots_figure is not None, "Should return a scatter plot figure"
