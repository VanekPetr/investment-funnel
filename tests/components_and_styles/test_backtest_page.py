"""
Tests for the backtest page components and functionality.

This module contains tests for the backtest_page.py module, which
provides the components and functionality for the backtesting page
of the investment funnel dashboard.
"""

import dash_bootstrap_components as dbc
from dash import html

from funnel.dashboard.components_and_styles.backtest_page import div


def test_div_creates_components(algo):
    """
    Test that div creates the expected components.

    This test verifies that the div function creates the expected
    components for the backtest page.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Call the function
    layout = div(algo)

    # Verify the results
    assert layout is not None, "Layout should be created"
    assert isinstance(layout, html.Div), "Layout should be a Div"

    # Check that the layout has the expected structure
    assert len(layout.children) == 1, "Layout should have one child"
    assert isinstance(layout.children[0], dbc.Row), "Layout child should be a Row"

    # Check that the row has the expected columns
    row = layout.children[0]
    assert len(row.children) == 3, "Row should have three columns"
    assert all(isinstance(col, dbc.Col) for col in row.children), "All row children should be Cols"
