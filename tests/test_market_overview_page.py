"""
Tests for the market overview page components and functionality.

This module contains tests for the market_overview_page.py module, which
provides the components and functionality for the market overview page
of the investment funnel dashboard.
"""

from unittest.mock import MagicMock, patch

import dash_bootstrap_components as dbc
import pytest
from dash import html
from dash.dcc import Graph
from dash.html import Div

from funnel.dashboard.components_and_styles.market_overview_page import (
    _divs,
    _generate_plot,
    _update_market_overview_plot,
    create_market_overview_layout,
    register_callbacks,
)


def test_generate_plot_valid_inputs(algo):
    """
    Test that _generate_plot works correctly with valid inputs.

    This test verifies that the _generate_plot function can generate a plot
    with valid inputs without raising any exceptions.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs
    start_date = algo.min_date
    end_date = algo.max_date
    fund_set = []
    top_performers = None
    top_performers_enabled = False
    top_performers_pct = 15

    # Call the function
    fig, top_performers_list = _generate_plot(
        algo,
        start_date,
        end_date,
        fund_set,
        top_performers,
        top_performers_enabled,
        top_performers_pct
    )

    # Verify the results
    assert fig is not None, "Figure should be created"
    assert isinstance(top_performers_list, list), "Top performers list should be a list"


def test_generate_plot_missing_dates(algo):
    """
    Test that _generate_plot raises an error when dates are missing.

    This test verifies that the _generate_plot function raises a ValueError
    when start_date or end_date is missing.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Test with missing start_date
    with pytest.raises(ValueError, match="Start date and end date must be provided"):
        _generate_plot(
            algo,
            start_date=None,
            end_date=algo.max_date,
            fund_set=[],
            top_performers=None,
            top_performers_enabled=False,
            top_performers_pct=15
        )

    # Test with missing end_date
    with pytest.raises(ValueError, match="Start date and end date must be provided"):
        _generate_plot(
            algo,
            start_date=algo.min_date,
            end_date=None,
            fund_set=[],
            top_performers=None,
            top_performers_enabled=False,
            top_performers_pct=15
        )


def test_generate_plot_invalid_percentage(algo):
    """
    Test that _generate_plot raises an error when top_performers_pct is invalid.

    This test verifies that the _generate_plot function raises a ValueError
    when top_performers_enabled is True and top_performers_pct is not between 0 and 100.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Test with negative percentage
    with pytest.raises(ValueError, match="Top performers percentage must be between 0 and 100"):
        _generate_plot(
            algo,
            start_date=algo.min_date,
            end_date=algo.max_date,
            fund_set=[],
            top_performers=None,
            top_performers_enabled=True,
            top_performers_pct=-5
        )

    # Test with percentage > 100
    with pytest.raises(ValueError, match="Top performers percentage must be between 0 and 100"):
        _generate_plot(
            algo,
            start_date=algo.min_date,
            end_date=algo.max_date,
            fund_set=[],
            top_performers=None,
            top_performers_enabled=True,
            top_performers_pct=105
        )


def test_generate_plot_error_handling(algo):
    """
    Test that _generate_plot handles errors correctly.

    This test verifies that the _generate_plot function handles errors
    from the algo.get_top_performing_assets and algo.plot_dots methods correctly.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Mock the algo.get_top_performing_assets method to raise an exception
    with patch.object(algo, 'get_top_performing_assets', side_effect=Exception("Test error")):
        with pytest.raises(ValueError, match="Failed to calculate top performing assets: Test error"):
            _generate_plot(
                algo,
                start_date=algo.min_date,
                end_date=algo.max_date,
                fund_set=[],
                top_performers=None,
                top_performers_enabled=True,
                top_performers_pct=15
            )

    # Mock the algo.plot_dots method to raise an exception
    with patch.object(algo, 'plot_dots', side_effect=Exception("Test error")):
        with pytest.raises(ValueError, match="Failed to generate plot: Test error"):
            _generate_plot(
                algo,
                start_date=algo.min_date,
                end_date=algo.max_date,
                fund_set=[],
                top_performers=None,
                top_performers_enabled=False,
                top_performers_pct=15
            )


def test_update_market_overview_plot_valid_inputs(algo):
    """
    Test that _update_market_overview_plot works correctly with valid inputs.

    This test verifies that the _update_market_overview_plot function can update
    the market overview plot with valid inputs without raising any exceptions.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs
    click = 1
    start = algo.min_date
    end = algo.max_date
    search = []
    top_performers = "no"
    top_performers_pct = 15

    # Call the function
    result = _update_market_overview_plot(
        algo,
        click,
        start,
        end,
        search,
        top_performers,
        top_performers_pct
    )

    # Verify the results
    assert isinstance(result, list), "Result should be a list"
    assert len(result) == 1, "Result should contain one element"
    assert isinstance(result[0], Graph), "Result should contain a Graph"


def test_update_market_overview_plot_no_click(algo):
    """
    Test that _update_market_overview_plot returns an empty div when click is None.

    This test verifies that the _update_market_overview_plot function returns
    an empty div when the click parameter is None or 0.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs
    click = None
    start = algo.min_date
    end = algo.max_date
    search = []
    top_performers = "no"
    top_performers_pct = 15

    # Call the function
    result = _update_market_overview_plot(
        algo,
        click,
        start,
        end,
        search,
        top_performers,
        top_performers_pct
    )

    # Verify the results
    assert isinstance(result, list), "Result should be a list"
    assert len(result) == 1, "Result should contain one element"
    assert isinstance(result[0], Div), "Result should contain a Div"


def test_update_market_overview_plot_missing_dates(algo):
    """
    Test that _update_market_overview_plot handles missing dates correctly.

    This test verifies that the _update_market_overview_plot function returns
    an error figure when start or end date is missing.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs
    click = 1
    start = None
    end = algo.max_date
    search = []
    top_performers = "no"
    top_performers_pct = 15

    # Call the function
    result = _update_market_overview_plot(
        algo,
        click,
        start,
        end,
        search,
        top_performers,
        top_performers_pct
    )

    # Verify the results
    assert isinstance(result, list), "Result should be a list"
    assert len(result) == 1, "Result should contain one element"
    assert isinstance(result[0], Graph), "Result should contain a Graph"
    assert "Error" in str(result[0].figure['layout']['title']), "Figure should have an error title"


def test_update_market_overview_plot_invalid_percentage(algo):
    """
    Test that _update_market_overview_plot handles invalid percentages correctly.

    This test verifies that the _update_market_overview_plot function returns
    an error figure when top_performers_pct is not between 0 and 100.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs
    click = 1
    start = algo.min_date
    end = algo.max_date
    search = []
    top_performers = "yes"
    top_performers_pct = -5

    # Call the function
    result = _update_market_overview_plot(
        algo,
        click,
        start,
        end,
        search,
        top_performers,
        top_performers_pct
    )

    # Verify the results
    assert isinstance(result, list), "Result should be a list"
    assert len(result) == 1, "Result should contain one element"
    assert isinstance(result[0], Graph), "Result should contain a Graph"
    assert "Error" in str(result[0].figure['layout']['title']), "Figure should have an error title"


def test_update_market_overview_plot_error_handling(algo):
    """
    Test that _update_market_overview_plot handles errors correctly.

    This test verifies that the _update_market_overview_plot function returns
    an error figure when an exception is raised during plot generation.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Mock the _generate_plot function to raise an exception
    with (patch('funnel.dashboard.components_and_styles.market_overview_page._generate_plot',
               side_effect=Exception("Test error"))):
        # Define test inputs
        click = 1
        start = algo.min_date
        end = algo.max_date
        search = []
        top_performers = "no"
        top_performers_pct = 15

        # Call the function
        result = _update_market_overview_plot(
            algo,
            click,
            start,
            end,
            search,
            top_performers,
            top_performers_pct
        )

        # Verify the results
        assert isinstance(result, list), "Result should be a list"
        assert len(result) == 1, "Result should contain one element"
        assert isinstance(result[0], Graph), "Result should contain a Graph"
        assert "Error" in str(result[0].figure['layout']['title']), "Figure should have an error title"
        assert "Test error" in str(result[0].figure['layout']['annotations'][0]['text'])

def test_divs_creates_components(algo):
    """
    Test that _divs creates the expected components.

    This test verifies that the _divs function creates the expected
    components for the market overview page.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Call the function
    components = _divs(algo)

    # Verify the results
    assert components is not None, "Components should be created"
    assert hasattr(components, 'optionsGraph'), "Components should have optionsGraph"
    assert hasattr(components, 'graphOverview'), "Components should have graphOverview"
    assert hasattr(components, 'spinner_dots'), "Components should have spinner_dots"

    # Check that the components have the expected structure
    assert isinstance(components.optionsGraph, html.Div), "optionsGraph should be a Div"
    assert isinstance(components.graphOverview, html.Div), "graphOverview should be a Div"
    assert isinstance(components.spinner_dots, html.Div), "spinner_dots should be a Div"


def test_divs_error_handling(algo):
    """
    Test that _divs handles errors correctly.

    This test verifies that the _divs function handles errors
    from the _generate_plot function correctly.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Mock the _generate_plot function to raise an exception
    with (patch('funnel.dashboard.components_and_styles.market_overview_page._generate_plot',
               side_effect=Exception("Test error"))):
        # Call the function
        components = _divs(algo)

        # Verify the results
        assert components is not None, "Components should be created even if plot generation fails"
        assert hasattr(components, 'optionsGraph'), "Components should have optionsGraph"
        assert hasattr(components, 'graphOverview'), "Components should have graphOverview"
        assert hasattr(components, 'spinner_dots'), "Components should have spinner_dots"

        # Check that the graphOverview contains an error figure
        assert isinstance(components.graphOverview, html.Div), "graphOverview should be a Div"
        assert "Error" in str(components.graphOverview.children.figure['layout']['title'])


def test_create_market_overview_layout(algo):
    """
    Test that create_market_overview_layout creates the expected layout.

    This test verifies that the create_market_overview_layout function creates
    the expected layout for the market overview page.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Call the function
    layout = create_market_overview_layout(algo)

    # Verify the results
    assert layout is not None, "Layout should be created"
    assert isinstance(layout, html.Div), "Layout should be a Div"

    # Check that the layout has the expected structure
    assert len(layout.children) == 1, "Layout should have one child"
    assert isinstance(layout.children[0], dbc.Row), "Layout child should be a Row"


def test_register_callbacks(algo):
    """
    Test that register_callbacks registers the expected callbacks.

    This test verifies that the register_callbacks function registers
    the expected callbacks for the market overview page.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Create a mock app
    app = MagicMock()

    # Call the function
    register_callbacks(app, algo)

    # Verify that the callbacks were registered
    assert app.callback.call_count == 2, "Two callbacks should be registered"
