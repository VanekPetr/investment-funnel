"""
Tests for the mobile page components and functionality.

This module contains tests for the mobile_page.py module, which
provides the components and functionality for the mobile page
of the investment funnel dashboard.
"""

import dash_bootstrap_components as dbc
from dash import html

from funnel.dashboard.components_and_styles.mobile_page import div


def test_div_creates_components(algo):
    """
    Test that div creates the expected components.

    This test verifies that the div function creates the expected
    components for the mobile page.

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

    # Check that the row has the expected content
    row = layout.children[0]
    assert len(row.children) == 1, "Row should have one child"
    assert isinstance(row.children[0], html.Div), "Row child should be a Div"

    # Check that the mobile page has the expected content
    mobile_page = row.children[0]
    assert len(mobile_page.children) == 3, "Mobile page should have three children"
    assert isinstance(mobile_page.children[0], html.Img), "First child should be an Img"
    assert isinstance(mobile_page.children[1], html.H1), "Second child should be an H1"
    assert isinstance(mobile_page.children[2], html.H4), "Third child should be an H4"
