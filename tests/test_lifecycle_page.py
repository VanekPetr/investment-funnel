"""
Tests for the lifecycle page components and functionality.

This module contains tests for the lifecycle_page.py module, which
provides the components and functionality for the lifecycle investments page
of the investment funnel dashboard.
"""

import dash_bootstrap_components as dbc
from dash import html

from funnel.dashboard.components_and_styles.lifecycle_page import (
    create_lifecycle_layout,
    div,
)


def test_div_creates_components(algo):
    """
    Test that div creates the expected components.

    This test verifies that the div function creates the expected
    components for the lifecycle page.

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


def test_create_lifecycle_layout(algo):
    """
    Test that create_lifecycle_layout creates the expected layout.

    This test verifies that the create_lifecycle_layout function creates
    the expected layout for the lifecycle page.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Call the function
    layout = create_lifecycle_layout(algo)

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
