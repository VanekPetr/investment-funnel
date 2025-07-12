from dataclasses import dataclass

from dash import dcc, html

from .components_and_styles.ai_feature_selection_page import create_ai_feature_selection_layout
from .components_and_styles.backtest_page import create_backtest_layout
from .components_and_styles.lifecycle_page import create_lifecycle_layout
from .components_and_styles.market_overview_page import create_market_overview_layout
from .components_and_styles.mobile import create_mobile_layout


@dataclass
class Layout:
    page_1: html.Div
    page_2: html.Div
    page_3: html.Div
    page_4: html.Div
    page_mobile: html.Div


def divs(algo) -> Layout:
    """
    Create all page layouts based on the provided algorithm instance.

    Args:
        algo: Algorithm-related object passed to layout builders.

    Returns:
        Layout: A named tuple of html.Div layouts.
    """
    return Layout(
        page_1=create_market_overview_layout(algo),
        page_2=create_ai_feature_selection_layout(algo),
        page_3=create_backtest_layout(algo),
        page_4=create_lifecycle_layout(algo),
        page_mobile=create_mobile_layout(algo)
    )


def load_page(page):
    """
    Load a page with the necessary components and data stores.

    Args:
        page: The page layout to load

    Returns:
        html.Div: The complete page layout with data stores
    """
    return html.Div(
        [
            # layout of the app
            dcc.Location(id="url"),
            html.Div(id="page-content", children=page),
        ]
    )
