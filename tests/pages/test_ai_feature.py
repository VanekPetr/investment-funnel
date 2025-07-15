"""
Tests for the ai_feature_selection page module.

This module contains tests for the ai_feature_selection.py module, which
provides the AI Feature Selection page for the investment funnel dashboard.
"""

import pytest
from unittest.mock import MagicMock, patch

import dash

# Mock dash.register_page before importing the module
dash.register_page = MagicMock()

from funnel.pages.ai_feature_selection import run_ml
from funnel.pages.models.ai_feature import FeatureInput, FeatureOutput


def test_run_ml_prevent_update():
    """
    Test that run_ml raises PreventUpdate when click is None or 0.

    This test verifies that the run_ml callback function correctly
    raises a PreventUpdate exception when the click parameter is falsy,
    preventing unnecessary updates to the UI.
    """
    with pytest.raises(dash.exceptions.PreventUpdate):
        run_ml(None, "MST", 4, "2013-01-01", "2014-01-01")

    with pytest.raises(dash.exceptions.PreventUpdate):
        run_ml(0, "MST", 4, "2013-01-01", "2014-01-01")


@patch("funnel.pages.ai_feature_selection.plot_ml")
def test_run_ml_processes_inputs(mock_plot_ml):
    """
    Test that run_ml correctly processes inputs and calls plot_ml.

    This test verifies that the run_ml callback function correctly
    processes the input parameters, creates a FeatureInput object,
    calls the plot_ml function with the correct arguments,
    and returns the outputs as a tuple.
    """
    # Create a mock FeatureOutput object
    mock_outputs = MagicMock(spec=FeatureOutput)
    mock_outputs.as_tuple.return_value = ("mock_figure", "mock_result", "mock_number")

    # Configure the mock plot_ml function to return the mock outputs
    mock_plot_ml.return_value = mock_outputs

    # Call the function being tested
    result = run_ml(1, "MST", 4, "2013-01-01", "2014-01-01")

    # Verify that plot_ml was called with the correct arguments
    mock_plot_ml.assert_called_once()

    # Get the arguments that plot_ml was called with
    args, kwargs = mock_plot_ml.call_args

    # Verify that the first argument is the algo object
    assert args[0] is not None

    # Verify that the second argument is a FeatureInput object with the correct values
    assert isinstance(args[1], FeatureInput)
    assert args[1].model == "MST"
    assert args[1].spec == 4
    assert args[1].start_date == "2013-01-01"
    assert args[1].end_date == "2014-01-01"

    # Verify that the result is the tuple returned by outputs.as_tuple()
    assert result == ("mock_figure", "mock_result", "mock_number")


def test_layout_components():
    """
    Test that the layout contains the expected components.

    This test verifies that the layout for the ai_feature_selection page
    contains the expected components with the correct properties.
    """
    from funnel.pages.ai_feature_selection import layout, optionML, graphML, spinner_ml
    import dash_bootstrap_components as dbc
    from dash import html

    # Verify that layout is a Div containing a Row
    assert isinstance(layout, html.Div)
    assert isinstance(layout.children, dbc.Row)

    # Verify that the Row contains two Cols
    row = layout.children
    assert len(row.children) == 2
    assert all(isinstance(col, dbc.Col) for col in row.children)

    # Verify that the first Col contains optionML
    assert row.children[0].children == optionML

    # Verify that the second Col contains graphML and spinner_ml
    assert row.children[1].children == [graphML, spinner_ml]