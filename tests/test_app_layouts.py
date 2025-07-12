"""
Tests for the app layouts.

This module contains tests for the app_layouts.py module, which
provides the layout functions for the different pages of the
investment funnel dashboard.
"""

from unittest.mock import patch

from dash import html

from funnel.dashboard.app_layouts import (
    page_1_layout,
    page_2_layout,
    page_3_layout,
    page_4_layout,
    page_mobile_layout,
)


def test_page_1_layout(algo):
    """
    Test that page_1_layout returns the expected layout.

    This test verifies that the page_1_layout function calls
    the market_overview_div function with the algo parameter
    and returns its result.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Create a mock return value
    mock_div = html.Div(id="mock-market-overview")

    # Mock the market_overview_div function
    with patch("funnel.dashboard.app_layouts.market_overview_div", return_value=mock_div) as mock_func:
        # Call the function
        result = page_1_layout(algo)

        # Verify the results
        assert result == mock_div, "page_1_layout should return the result of market_overview_div"
        mock_func.assert_called_once_with(algo), "market_overview_div should be called with algo"


def test_page_2_layout(algo):
    """
    Test that page_2_layout returns the expected layout.

    This test verifies that the page_2_layout function calls
    the ai_feature_selection_div function with the algo parameter
    and returns its result.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Create a mock return value
    mock_div = html.Div(id="mock-ai-feature-selection")

    # Mock the ai_feature_selection_div function
    with patch("funnel.dashboard.app_layouts.ai_feature_selection_div", return_value=mock_div) as mock_func:
        # Call the function
        result = page_2_layout(algo)

        # Verify the results
        assert result == mock_div, "page_2_layout should return the result of ai_feature_selection_div"
        mock_func.assert_called_once_with(algo), "ai_feature_selection_div should be called with algo"


def test_page_3_layout(algo):
    """
    Test that page_3_layout returns the expected layout.

    This test verifies that the page_3_layout function calls
    the backtest_div function with the algo parameter
    and returns its result.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Create a mock return value
    mock_div = html.Div(id="mock-backtest")

    # Mock the backtest_div function
    with patch("funnel.dashboard.app_layouts.backtest_div", return_value=mock_div) as mock_func:
        # Call the function
        result = page_3_layout(algo)

        # Verify the results
        assert result == mock_div, "page_3_layout should return the result of backtest_div"
        mock_func.assert_called_once_with(algo), "backtest_div should be called with algo"


def test_page_4_layout(algo):
    """
    Test that page_4_layout returns the expected layout.

    This test verifies that the page_4_layout function calls
    the lifecycle_div function with the algo parameter
    and returns its result.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Create a mock return value
    mock_div = html.Div(id="mock-lifecycle")

    # Mock the lifecycle_div function
    with patch("funnel.dashboard.app_layouts.lifecycle_div", return_value=mock_div) as mock_func:
        # Call the function
        result = page_4_layout(algo)

        # Verify the results
        assert result == mock_div, "page_4_layout should return the result of lifecycle_div"
        mock_func.assert_called_once_with(algo), "lifecycle_div should be called with algo"


def test_page_mobile_layout(algo):
    """
    Test that page_mobile_layout returns the expected layout.

    This test verifies that the page_mobile_layout function calls
    the mobile_page_div function with the algo parameter
    and returns its result.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Create a mock return value
    mock_div = html.Div(id="mock-mobile-page")

    # Mock the mobile_page_div function
    with patch("funnel.dashboard.app_layouts.mobile_page_div", return_value=mock_div) as mock_func:
        # Call the function
        result = page_mobile_layout(algo)

        # Verify the results
        assert result == mock_div, "page_mobile_layout should return the result of mobile_page_div"
        mock_func.assert_called_once_with(algo), "mobile_page_div should be called with algo"
