from typing import Any

from funnel.dashboard.app_layouts import divs as app_layouts
from funnel.dashboard.app_layouts import load_page
from funnel.dashboard.components_and_styles.ai_feature_selection_page import divs as ai_feature_selection
from funnel.dashboard.components_and_styles.backtest_page import divs as backtest
from funnel.dashboard.components_and_styles.lifecycle_page import divs as lifecycle
from funnel.dashboard.components_and_styles.market_overview_page import divs as market_overview


def test_marketOverview(algo: Any) -> None:
    """
    Test that the market overview page components can be created.

    This test verifies that the market overview page components can be
    successfully created using the market_overview function without
    raising any exceptions.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Create the market overview components
    components = market_overview(algo)

    # Verify that components were created
    assert components is not None, "Market overview components should be created"


def test_lifecycle(algo: Any) -> None:
    """
    Test that the lifecycle page components can be created.

    This test verifies that the lifecycle page components can be
    successfully created using the lifecycle function without
    raising any exceptions.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Create the lifecycle components
    components = lifecycle(algo)

    # Verify that components were created
    assert components is not None, "Lifecycle components should be created"


def test_backtest(algo: Any) -> None:
    """
    Test that the backtest page components can be created.

    This test verifies that the backtest page components can be
    successfully created using the backtest function without
    raising any exceptions.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Create the backtest components
    components = backtest(algo)

    # Verify that components were created
    assert components is not None, "Backtest components should be created"


def test_ai_feature_selection(algo: Any) -> None:
    """
    Test that the AI feature selection page components can be created.

    This test verifies that the AI feature selection page components can be
    successfully created using the ai_feature_selection function without
    raising any exceptions.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Create the AI feature selection components
    components = ai_feature_selection(algo)

    # Verify that components were created
    assert components is not None, "AI feature selection components should be created"


def test_layouts(algo: Any) -> None:
    """
    Test that the complete layout can be created.

    This test verifies that the complete application layout can be
    successfully created using the app_layouts function without
    raising any exceptions.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Create the complete layout
    layout = app_layouts(algo)

    # Verify that layout was created
    assert layout is not None, "Complete layout should be created"


def test_load_page(algo: Any) -> None:
    """
    Test that a page can be loaded with the load_page function.

    This test verifies that the load_page function can successfully
    load a page layout with all necessary components and data stores
    without raising any exceptions.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Create the layout
    layout = app_layouts(algo)

    # Load the market overview page
    page = load_page(layout.page_1, algo)

    # Verify that page was loaded (implicitly checks that no exception was raised)
    assert page is not None, "Page should be loaded successfully"
