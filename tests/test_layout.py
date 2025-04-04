from funnel.dashboard.app_layouts import divs as app_layouts
from funnel.dashboard.app_layouts import load_page
from funnel.dashboard.components_and_styles.ai_feature_selection_page import divs as ai_feature_selection
from funnel.dashboard.components_and_styles.backtest_page import divs as backtest
from funnel.dashboard.components_and_styles.lifecycle_page import divs as lifecycle
from funnel.dashboard.components_and_styles.market_overview_page import divs as market_overview


def test_marketOverview(algo):
    x = market_overview(algo)
    assert x is not None

def test_lifecycle(algo):
    x = lifecycle(algo)
    assert x is not None

def test_backtest(algo):
    x = backtest(algo)
    assert x is not None

def test_ai_feature_selection(algo):
    x = ai_feature_selection(algo)
    assert x is not None

def test_layouts(algo):
    x = app_layouts(algo)
    assert x is not None

def test_load_page(algo):
    x = app_layouts(algo)
    load_page(x.page_1, algo)
