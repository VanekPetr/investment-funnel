"""
Tests for the app_callbacks module.

This module contains tests for the app_callbacks.py module, which
provides the callback functions for the investment funnel dashboard.
"""

from funnel.models.ai_feature import FeatureInput, FeatureOutput, plot_ml
from funnel.models.backtest import BacktestInputs, BacktestOutputs, plot_backtest
from funnel.models.lifecycle import LifecycleInputs, LifecycleOutputs, plot_lifecycle
from funnel.models.overview import OverviewInputs, plot_overview, OverviewOutputs


def test_plot_lifecycle(algo):
    """
    Test that plot_lifecycle generates new figures when clicked.

    This test verifies that the plot_lifecycle function generates
    new figures when the button is clicked.

    Args:
        callbacks: The callbacks fixture
    """
    # Call the function
    inputs = LifecycleInputs(
        model="MST",  # model
        model_spec=2,  # model_spec
        pick_top=3,  # pick_top
        scen_model="Bootstrap",  # scen_model
        scen_spec=200,  # scen_spec
        start_date="2019-01-01",  # start_data
        end_date="2020-01-01",  # end_train
        end_year=2030,  # end_year
        portfolio_value=100000,  # portfolio_value
        yearly_withdraws=1000,  # yearly_withdraws
        risk_preference=15,  # risk_preference
    )

    result = plot_lifecycle(
        algo,  # algo object
        inputs
    )

    assert isinstance(result, LifecycleOutputs)

    # Verify the results
    assert result.glidepaths_output_fig is not None, "Should return a glidepaths figure"
    assert result.performance_output_fig is not None, "Should return a performance figure"
    assert result.lifecycle_all_output_fig is not None, "Should return a lifecycle all figure"


def test_plot_backtest(algo):
    """
    Test that plot_backtest generates new figures when clicked.

    This test verifies that the plot_backtest function generates
    new figures when the button is clicked.

    Args:
        callbacks: The callbacks fixture
    """

    inputs = BacktestInputs(
        start_train_date="2018-01-01",
        end_train_date="2021-12-31",
        start_test_date="2022-01-01",
        end_test_date="2023-01-01",
        model="MST",
        model_spec=2,
        pick_top=5,
        scen_model="Bootstrap",
        scen_spec=1000,
        benchmark=["Wealth Invest Amalie Global AK"],
        solver="CLARABEL",
        optimization_model="CVaR model",
        lower_bound=0
    )

    result = plot_backtest(
        algo,  # algo object
        inputs
    )

    assert isinstance(result, BacktestOutputs)



def test_plot_ml(algo):
    """
    Test that plot_ml generates new figures when clicked.

    This test verifies that the plot_ml function generates
    new figures when the button is clicked.

    Args:
        callbacks: The callbacks fixture
    """
    # Get the plot_ml function
    inputs = FeatureInput(
        start_date="2013-01-01",
        end_date="2014-01-01",
        model="MST",
        spec=4
    )

    result = plot_ml(
        algo,  # algo object
        inputs
    )

    assert isinstance(result, FeatureOutput)
    assert result.ml_figure is not None, "Should return a ml figure"


def test_plot_overview(algo):
    """
    Test that plot_overview generates new figures when clicked.

    This test verifies that the plot_overview function generates
    new figures when the button is clicked.

    Args:
        callbacks: The callbacks fixture
    """
    inputs = OverviewInputs(
        start_date="2013-01-01",
        end_date="2023-01-01",
        search=["Wealth Invest Amalie Global AK", "BankInvest Danske Aktier A"],
        top_performers="yes"
    )

    result = plot_overview(
        algo,  # algo object
        inputs
    )

    assert isinstance(result, OverviewOutputs)
    assert result.dots_figure is not None, "Should return a glidepaths figure"
