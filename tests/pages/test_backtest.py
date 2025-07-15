"""
Tests for the backtest page module.

This module contains tests for the backtest.py module, which
provides the Backtesting page for the investment funnel dashboard.
"""

from unittest.mock import MagicMock, patch

import dash
import pytest

# Mock dash.register_page before importing the module
dash.register_page = MagicMock()

from funnel.pages.backtest import run_backtest
from funnel.pages.models.backtest import BacktestInputs, BacktestOutputs


def test_run_backtest_prevent_update():
    """
    Test that run_backtest raises PreventUpdate when click is None or 0.

    This test verifies that the run_backtest callback function correctly
    raises a PreventUpdate exception when the click parameter is falsy,
    preventing unnecessary updates to the UI.
    """
    with pytest.raises(dash.exceptions.PreventUpdate):
        run_backtest(None, "MST", 2, 3, "Bootstrap", 1000, ["Benchmark"], "2018-01-01", "2021-12-31",
                    "2022-01-01", "2023-01-01", "CLARABEL", "CVaR model", 0)

    with pytest.raises(dash.exceptions.PreventUpdate):
        run_backtest(0, "MST", 2, 3, "Bootstrap", 1000, ["Benchmark"], "2018-01-01", "2021-12-31",
                    "2022-01-01", "2023-01-01", "CLARABEL", "CVaR model", 0)


@patch("funnel.pages.backtest.plot_backtest")
def test_run_backtest_processes_inputs(mock_plot_backtest):
    """
    Test that run_backtest correctly processes inputs and calls plot_backtest.

    This test verifies that the run_backtest callback function correctly
    processes the input parameters, creates a BacktestInputs object,
    calls the plot_backtest function with the correct arguments,
    and returns the outputs as a tuple.
    """
    # Create a mock BacktestOutputs object
    mock_outputs = MagicMock(spec=BacktestOutputs)
    mock_outputs.as_tuple.return_value = ("mock_perf", "mock_comp", "mock_universe")

    # Configure the mock plot_backtest function to return the mock outputs
    mock_plot_backtest.return_value = mock_outputs

    # Call the function being tested
    result = run_backtest(1, "MST", 2, 3, "Bootstrap", 1000, ["Benchmark"], "2018-01-01", "2021-12-31",
                         "2022-01-01", "2023-01-01", "CLARABEL", "CVaR model", 0)

    # Verify that plot_backtest was called with the correct arguments
    mock_plot_backtest.assert_called_once()

    # Get the arguments that plot_backtest was called with
    args, kwargs = mock_plot_backtest.call_args

    # Verify that the first argument is the algo object
    assert args[0] is not None

    # Verify that the second argument is a BacktestInputs object with the correct values
    assert isinstance(args[1], BacktestInputs)
    assert args[1].model == "MST"
    assert args[1].model_spec == 2
    assert args[1].pick_top == 3
    assert args[1].scen_model == "Bootstrap"
    assert args[1].scen_spec == 1000
    assert args[1].benchmark == ["Benchmark"]
    assert args[1].start_train_date == "2018-01-01"
    assert args[1].end_train_date == "2021-12-31"
    assert args[1].start_test_date == "2022-01-01"
    assert args[1].end_test_date == "2023-01-01"
    assert args[1].solver == "CLARABEL"
    assert args[1].optimization_model == "CVaR model"
    assert args[1].lower_bound == 0

    # Verify that the result is the tuple returned by outputs.as_tuple()
    assert result == ("mock_perf", "mock_comp", "mock_universe")


def test_layout_components():
    """
    Test that the layout contains the expected components.

    This test verifies that the layout for the backtest page
    contains the expected components with the correct properties.
    """
    import dash_bootstrap_components as dbc
    from dash import html

    from funnel.pages.backtest import graphResults, layout, optionBacktest, spinner_backtest

    # Verify that layout is a Div containing a Row
    assert isinstance(layout, html.Div)
    assert isinstance(layout.children, dbc.Row)

    # Verify that the Row contains two Cols
    row = layout.children
    assert len(row.children) == 2
    assert all(isinstance(col, dbc.Col) for col in row.children)

    # Verify that the first Col contains optionBacktest
    assert row.children[0].children == optionBacktest

    # Verify that the second Col contains graphResults and spinner_backtest
    assert row.children[1].children == [graphResults, spinner_backtest]
