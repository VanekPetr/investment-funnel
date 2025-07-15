"""
Tests for the overview page module.

This module contains tests for the overview.py module, which
provides the Market Overview page for the investment funnel dashboard.
"""

from unittest.mock import MagicMock, patch

import dash
import pytest

# Mock dash.register_page before importing the module
dash.register_page = MagicMock()

from funnel.pages.models.overview import OverviewInputs, OverviewOutputs
from funnel.pages.overview import plot_dots


def test_plot_dots_prevent_update():
    """
    Test that plot_dots raises PreventUpdate when click is None or 0.

    This test verifies that the plot_dots callback function correctly
    raises a PreventUpdate exception when the click parameter is falsy,
    preventing unnecessary updates to the UI.
    """
    with pytest.raises(dash.exceptions.PreventUpdate):
        plot_dots(None, "2020-01-01", "2021-01-01", [], "no")

    with pytest.raises(dash.exceptions.PreventUpdate):
        plot_dots(0, "2020-01-01", "2021-01-01", [], "no")


@patch("funnel.pages.overview.plot_overview")
def test_plot_dots_processes_inputs(mock_plot_overview):
    """
    Test that plot_dots correctly processes inputs and calls plot_overview.

    This test verifies that the plot_dots callback function correctly
    processes the input parameters, creates an OverviewInputs object,
    calls the plot_overview function with the correct arguments,
    and returns the outputs as a tuple.
    """
    # Create a mock OverviewOutputs object
    mock_outputs = MagicMock(spec=OverviewOutputs)
    mock_outputs.as_tuple.return_value = ("mock_figure",)

    # Configure the mock plot_overview function to return the mock outputs
    mock_plot_overview.return_value = mock_outputs

    # Call the function being tested
    result = plot_dots(1, "2020-01-01", "2021-01-01", ["Fund1", "Fund2"], "yes")

    # Verify that plot_overview was called with the correct arguments
    mock_plot_overview.assert_called_once()

    # Get the arguments that plot_overview was called with
    args, kwargs = mock_plot_overview.call_args

    # Verify that the first argument is the algo object
    assert args[0] is not None

    # Verify that the second argument is an OverviewInputs object with the correct values
    assert isinstance(args[1], OverviewInputs)
    assert args[1].start_date == "2020-01-01"
    assert args[1].end_date == "2021-01-01"
    assert args[1].search == ["Fund1", "Fund2"]
    assert args[1].top_performers == "yes"

    # Verify that the result is the tuple returned by outputs.as_tuple()
    assert result == ("mock_figure",)


def test_layout_components():
    """
    Test that the layout contains the expected components.

    This test verifies that the layout for the overview page
    contains the expected components with the correct properties.
    """
    import dash_bootstrap_components as dbc
    from dash import html

    from funnel.pages.overview import graphOverview, layout, optionsGraph, spinner_dots

    # Verify that layout is a Div containing a Row
    assert isinstance(layout, html.Div)
    assert isinstance(layout.children, dbc.Row)

    # Verify that the Row contains two Cols
    row = layout.children
    assert len(row.children) == 2
    assert all(isinstance(col, dbc.Col) for col in row.children)

    # Verify that the first Col contains optionsGraph
    assert row.children[0].children == optionsGraph

    # Verify that the second Col contains graphOverview and spinner_dots
    assert row.children[1].children == [graphOverview, spinner_dots]
