from funnel.dashboard.components_and_styles.market_overview_page import divs


def test_marketOverview(algo):
    x = divs(algo)
    assert x is not None
