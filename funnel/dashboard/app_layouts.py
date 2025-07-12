from typing import NamedTuple

from dash import dcc, html

from .components_and_styles.ai_feature_selection_page import create_ai_feature_selection_layout
from .components_and_styles.backtest_page import create_backtest_layout
from .components_and_styles.lifecycle_page import create_lifecycle_layout
from .components_and_styles.market_overview_page import create_market_overview_layout
from .components_and_styles.mobile import create_mobile_layout

Layout = NamedTuple('Layout', [
    ('page_1', html.Div),
    ('page_2', html.Div),
    ('page_3', html.Div),
    ('page_4', html.Div),
    ('page_mobile', html.Div),
])


def divs(algo) -> Layout:
    """Create the layouts for all pages."""
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
        algo: The algorithm object containing data and methods

    Returns:
        html.Div: The complete page layout with data stores
    """
    return html.Div(
        [
            # layout of the app
            dcc.Location(id="url"),
            html.Div(id="page-content", children=page),
            # # Hidden divs to store data
            # dcc.Store(id="saved-start-date-page-0", data=algo.min_date),
            # dcc.Store(id="saved-end-date-page-0", data=algo.max_date),
            # dcc.Store(id="saved-find-fund", data=[]),
            # dcc.Store(id="saved-top-performers-names", data=[]),
            # dcc.Store(id="saved-top-performers", data="no"),
            # dcc.Store(id="saved-combine-top-performers", data="no"),
            # dcc.Store(id="saved-top-performers-pct", data=15),
            # dcc.Store(id="saved-figure-page-0", data=None),
            # dcc.Store(id="saved-start-date-page-1", data=algo.min_date),
            # dcc.Store(id="saved-end-date-page-1", data=algo.max_date),
            # dcc.Store(id="saved-ml-model", data=""),
            # dcc.Store(id="saved-ml-spec", data=""),
            # dcc.Store(id="saved-ml-text", data="No selected asset."),
            # dcc.Store(id="saved-figure-page-1", data=None),
            # dcc.Store(
            #     id="saved-ai-table",
            #     data=pd.DataFrame(
            #         np.array(
            #             [
            #                 [
            #                     "No result",
            #                     "No result",
            #                     "No result",
            #                     "No result",
            #                     "No result",
            #                 ]
            #             ]
            #         ),
            #         columns=[
            #             "Name",
            #             "ISIN",
            #             "Sharpe Ratio",
            #             "Average Annual Returns",
            #             "Standard Deviation of Returns",
            #         ],
            #     ).to_dict("records"),
            # ),
            # dcc.Store(id="saved-split-date", data=algo.min_date),
            # dcc.Store(id="saved-ml-model-back", data=""),
            # dcc.Store(id="saved-ml-spec-back", data=2),
            # dcc.Store(id="saved-pick-num-back", data=5),
            # dcc.Store(id="saved-scen-model-back", data=""),
            # dcc.Store(id="saved-scen-spec-back", data=1000),
            # dcc.Store(id="saved-benchmark-back", data=[]),
            # dcc.Store(id="saved-perf-figure-page-2", data=None),
            # dcc.Store(id="saved-comp-figure-page-2", data=None),
            # dcc.Store(id="saved-universe-figure-page-2", data=None),
            # dcc.Store(id="saved-solver", data=""),
            # dcc.Store(id="saved-optimization-model", data=""),
            # dcc.Store(id="saved-ml-model-lifecycle", data=""),
            # dcc.Store(id="saved-ml-spec-lifecycle", data=2),
            # dcc.Store(id="saved-pick-num-lifecycle", data=5),
            # dcc.Store(id="saved-scen-model-lifecycle", data=""),
            # dcc.Store(id="saved-scen-spec-lifecycle", data=1000),
            # dcc.Store(id="saved-glidepaths-figure-page-3", data=None),
            # dcc.Store(id="saved-performance-figure-page-3", data=None),
            # dcc.Store(id="saved-lifecycle-all-figure-page-3", data=None),
            # dcc.Store(id="saved_portfolio_value", data=None),
            # dcc.Store(id="saved_yearly_withdraws", data=None),
            # dcc.Store(id="saved_risk_preference", data=None),
            # dcc.Store(id="saved_end_year", data=2040),
        ]
    )
