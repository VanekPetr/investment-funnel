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
