"""
Tests for the lifecycle page module.

This module contains tests for the lifecycle.py module, which
provides the Lifecycle Investment page for the investment funnel dashboard.
"""

from unittest.mock import MagicMock, patch

import dash
import pytest

from funnel.pages.lifecycle import run_lifecycle
from funnel.pages.models.lifecycle import LifecycleInputs, LifecycleOutputs


def test_run_lifecycle_prevent_update():
    """
    Test that run_lifecycle raises PreventUpdate when click is None or 0.

    This test verifies that the run_lifecycle callback function correctly
    raises a PreventUpdate exception when the click parameter is falsy,
    preventing unnecessary updates to the UI.
    """
    with pytest.raises(dash.exceptions.PreventUpdate):
        run_lifecycle(None, "MST", 2, 3, "Bootstrap", 200, "2019-01-01", "2020-01-01", 2030, 100000, 1000, 15)

    with pytest.raises(dash.exceptions.PreventUpdate):
        run_lifecycle(0, "MST", 2, 3, "Bootstrap", 200, "2019-01-01", "2020-01-01", 2030, 100000, 1000, 15)


@patch("funnel.pages.lifecycle.plot_lifecycle")
def test_run_lifecycle_processes_inputs(mock_plot_lifecycle):
    """
    Test that run_lifecycle correctly processes inputs and calls plot_lifecycle.

    This test verifies that the run_lifecycle callback function correctly
    processes the input parameters, creates a LifecycleInputs object,
    calls the plot_lifecycle function with the correct arguments,
    and returns the outputs as a tuple.
    """
    # Create a mock LifecycleOutputs object
    mock_outputs = MagicMock(spec=LifecycleOutputs)
    mock_outputs.as_tuple.return_value = ("mock_glidepaths", "mock_performance", "mock_lifecycle_all")

    # Configure the mock plot_lifecycle function to return the mock outputs
    mock_plot_lifecycle.return_value = mock_outputs

    # Call the function being tested
    result = run_lifecycle(1, "MST", 2, 3, "Bootstrap", 200, "2019-01-01", "2020-01-01", 2030, 100000, 1000, 15)

    # Verify that plot_lifecycle was called with the correct arguments
    mock_plot_lifecycle.assert_called_once()

    # Get the arguments that plot_lifecycle was called with
    args, kwargs = mock_plot_lifecycle.call_args

    # Verify that the first argument is the algo object
    assert args[0] is not None

    # Verify that the second argument is a LifecycleInputs object with the correct values
    assert isinstance(args[1], LifecycleInputs)
    assert args[1].model == "MST"
    assert args[1].model_spec == 2
    assert args[1].pick_top == 3
    assert args[1].scen_model == "Bootstrap"
    assert args[1].scen_spec == 200
    assert args[1].start_date == "2019-01-01"
    assert args[1].end_date == "2020-01-01"
    assert args[1].end_year == 2030
    assert args[1].portfolio_value == 100000
    assert args[1].yearly_withdraws == 1000
    assert args[1].risk_preference == 15

    # Verify that the result is the tuple returned by outputs.as_tuple()
    assert result == ("mock_glidepaths", "mock_performance", "mock_lifecycle_all")


def test_layout_components():
    """
    Test that the layout contains the expected components.

    This test verifies that the layout for the lifecycle page
    contains the expected components with the correct properties.
    """
    import dash_bootstrap_components as dbc
    from dash import html

    from funnel.pages.lifecycle import layout, options_lifecycle, results_lifecycle, spinner_lifecycle

    # Verify that layout is a Div containing a Row
    assert isinstance(layout, html.Div)
    assert isinstance(layout.children, dbc.Row)

    # Verify that the Row contains two Cols
    row = layout.children
    assert len(row.children) == 2
    assert all(isinstance(col, dbc.Col) for col in row.children)

    # Verify that the first Col contains options_lifecycle
    assert row.children[0].children == options_lifecycle

    # Verify that the second Col contains results_lifecycle and spinner_lifecycle
    assert row.children[1].children == [results_lifecycle, spinner_lifecycle]
