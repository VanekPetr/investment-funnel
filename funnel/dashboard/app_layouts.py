from .components_and_styles.ai_feature_selection_page import div as ai_feature_selection_div
from .components_and_styles.backtest_page import div as backtest_div
from .components_and_styles.lifecycle_page import div as lifecycle_div
from .components_and_styles.market_overview_page import div as market_overview_div
from .components_and_styles.mobile_page import div as mobile_page_div


# *** LIFECYCLE ***
def page_4_layout(algo):
    """
    Create the layout for the Lifecycle Investments page.

    Args:
        algo: The algorithm object containing data and methods

    Returns:
        html.Div: The layout for the Lifecycle Investments page
    """
    return lifecycle_div(algo)

# *** BACK-TESTING ***
def page_3_layout(algo):
    """
    Create the layout for the Backtesting page.

    Args:
        algo: The algorithm object containing data and methods

    Returns:
        html.Div: The layout for the Backtesting page
    """
    return backtest_div(algo)

# *** AI Feature Selection ***
def page_2_layout(algo):
    """
    Create the layout for the AI Feature Selection page.

    Args:
        algo: The algorithm object containing data and methods

    Returns:
        html.Div: The layout for the AI Feature Selection page
    """
    return ai_feature_selection_div(algo)

# *** MARKET OVERVIEW ***
def page_1_layout(algo):
    """
    Create the layout for the Market Overview page.

    Args:
        algo: The algorithm object containing data and methods

    Returns:
        html.Div: The layout for the Market Overview page
    """
    return market_overview_div(algo)

# *** MOBILE PAGE ***
def page_mobile_layout(algo):
    """
    Create the layout for the mobile page.

    Args:
        algo: The algorithm object containing data and methods

    Returns:
        html.Div: The layout for the mobile page
    """
    return mobile_page_div(algo)
