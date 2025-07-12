import logging

import flask
from dash.dependencies import Input, Output

# Set up logging
logger = logging.getLogger(__name__)

def get_callbacks(app, layout, algo):
    """
    Register all callbacks for the dashboard application.

    Args:
        app: The Dash application instance
        layout: The layout object containing page layouts
        algo: The algorithm object containing data and methods
    """

    # NAVIGATION
    # -----------------------------------------------------------------------------------------------------------------
    @app.callback(Output("page-content", "children"), [Input("url", "pathname")])
    def display_page(pathname: str):
        """
        Display the appropriate page based on the URL pathname.
        Detects mobile devices and shows the mobile page if necessary.

        Args:
            pathname: The URL pathname

        Returns:
            The appropriate page layout
        """
        is_mobile = flask.request.headers.get("User-Agent").lower()
        if "mobile" in is_mobile or "mobi" in is_mobile:
            return layout.page_mobile
        elif pathname == "/" or pathname == "":
            return layout.page_1
        elif pathname == "/page-1":
            return layout.page_2
        elif pathname == "/page-2":
            return layout.page_3
        else:
            return layout.page_4


    # LIFECYCLE INVESTMENT ANALYSIS
    # -----------------------------------------------------------------------------------------------------------------
    # Import and register the lifecycle callbacks
    from .components_and_styles.lifecycle_page import register_callbacks as reg_lifecycle

    # Register the lifecycle callbacks
    reg_lifecycle(app, algo)


    # BACK-TESTING
    # -----------------------------------------------------------------------------------------------------------------
    # Import and register the backtest callbacks
    from .components_and_styles.backtest_page import register_callbacks as reg_backtest

    # Register the backtest callbacks
    reg_backtest(app, algo)


    # AI Feature Selection
    # -----------------------------------------------------------------------------------------------------------------
    # Import and register the AI feature selection callbacks

    # from .components_and_styles.ai_feature_selection_page import register_callbacks as reg_ai_feature_selection

    # Register the AI feature selection callbacks
    # reg_ai_feature_selection(app, algo)


    # MARKET OVERVIEW
    # -----------------------------------------------------------------------------------------------------------------
    # Import and register the market overview callbacks
    from .components_and_styles.market_overview_page import register_callbacks as reg_market_overview

    # Register the market overview callbacks
    reg_market_overview(app, algo)
