"""
Tests for the model functions in the pages/models directory.

This module contains tests for the model functions that power the dashboard pages.
Each test verifies that a specific model function correctly processes input data
and generates the expected output visualizations.
"""

from typing import Any

from funnel.pages.models.ai_feature import FeatureInput, FeatureOutput, plot_ml
from funnel.pages.models.backtest import BacktestInputs, BacktestOutputs, plot_backtest
from funnel.pages.models.lifecycle import LifecycleInputs, LifecycleOutputs, plot_lifecycle
from funnel.pages.models.overview import OverviewInputs, OverviewOutputs, plot_overview


def test_plot_lifecycle(algo: Any) -> None:
    """
    Test that plot_lifecycle generates new figures when clicked.

    This test verifies that the plot_lifecycle function correctly processes
    input parameters and generates the expected visualization outputs for
    lifecycle investment analysis.

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


def test_plot_backtest(algo: Any) -> None:
    """
    Test that plot_backtest generates new figures when clicked.

    This test verifies that the plot_backtest function correctly processes
    input parameters and generates the expected visualization outputs for
    backtesting investment strategies.

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


def test_plot_overview(algo: Any) -> None:
    """
    Test that plot_overview generates new figures when clicked.

    This test verifies that the plot_overview function correctly processes
    input parameters and generates the expected visualization outputs for
    the market overview.

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
