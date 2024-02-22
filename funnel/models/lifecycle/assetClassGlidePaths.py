from typing import List

import pandas as pd
from loguru import logger

# -----------------------------------------------------------------------------------------------------------

# This code is still under development, and is not to be a part of the investment funnel set-up yet.

# -----------------------------------------------------------------------------------------------------------


def create_asset_class_df(tickers: List[str]) -> pd.DataFrame:
    """
    Create an asset classification DataFrame with an optional "Cash" entry.

    Parameters:
    - asset_tickers: List of asset identifiers (e.g., ISIN codes).
    - asset_classes: List of classifications ('risky' or 'non_risky') for each asset.

    Returns:
    - A pandas DataFrame with asset identifiers as index and their classifications.
    """

    """
    Old MST run with 19 assets:
    # Define the tickers and their corresponding asset classes
    ticker = ['LU0368260294', 'DK0060188902', 'DK0010264456', 'DK0015942650', 'LU0332084994',
              'GB00B0XNFF59', 'LU0193801577', 'DK0016272602', 'DK0060118610', 'DK0060238194',
              'DK0016262728', 'DK0060037455', 'LU1028171848', 'IE00B1XNHC34', 'IE00B5WHFQ43',
              'IE00BM67HQ30', 'DE000A0H08S0', 'IE00B42Z5J44', 'IE00B0M63391']
    asset_classes = ['risky', 'risky', 'risky', 'non_risky', 'non_risky', 'non_risky', 'risky',
                     'risky', 'non_risky', 'non_risky', 'non_risky', 'non_risky', 'risky', 'risky',
                     'risky', 'risky', 'risky', 'risky', 'risky']

    # Create a DataFrame from the lists
    df = pd.DataFrame({
        'Ticker': ticker,
        'Asset Class': asset_classes
    })
    """
    """
    ticker = ['LU0368260294', 'DK0060188902', 'DK0060244242', 'DK0010264456', 'DK0015942650', 'LU0332084994',
              'LU0012195615', 'LU0123485178', 'LU0193801577', 'DK0016272602', 'DK0016205255', 'LU0244270301',
              'DK0060238194', 'DK0016261910', 'DK0016262728', 'DK0060037455', 'LU0705259686', 'DK0060300929',
              'DK0060009249', 'DK0060820173', 'DK0060498269', 'IE00B1XNHC34', 'DE000A0Q4R36', 'IE00B14X4T88',
              'IE00B27YCF74', 'LU0456910800', 'IE00B5WHFQ43', 'IE00BM67HQ30', 'IE00BM67HR47', 'DE000A0H08S0',
              'IE00B42Z5J44', 'IE00B0M63391', 'DE000A0H08K7', 'IE00B6R52143', 'LU0290357929']
    """
    asset_classes = [
        "risky",
        "risky",
        "risky",
        "risky",
        "non_risky",
        "non_risky",
        "risky",
        "risky",
        "risky",
        "risky",
        "non_risky",
        "risky",
        "non_risky",
        "non_risky",
        "non_risky",
        "non_risky",
        "non_risky",
        "non_risky",
        "non_risky",
        "risky",
        "risky",
        "risky",
        "risky",
        "risky",
        "risky",
        "risky",
        "risky",
        "risky",
        "risky",
        "risky",
        "risky",
        "risky",
        "risky",
        "risky",
        "non_risky",
    ]

    asset_tickers = tickers.copy()
    # Ensure "Cash" is included only if it's not already in the list
    if "Cash" not in asset_tickers:
        asset_tickers.append("Cash")
        asset_classes.append(
            "non_risky"
        )  # Assuming "Cash" is classified as 'non_risky'

    # Create the DataFrame
    asset_class_df = pd.DataFrame(
        asset_classes, index=asset_tickers, columns=["Asset Class"]
    )

    return asset_class_df


def calculate_target_allocation(
    annual_alloc_targets: pd.DataFrame, tickers: List[str]
) -> pd.DataFrame:
    # Check if the length of the tickers list is not 19
    if len(tickers) != 35:
        logger.debug(
            "Right now we work with a special amount of Minimum Spanning Tree runs. Please choose 5 MST "
            "runs such that we work with the 35 assets that we have already determined asset class for. "
            "If we still see an error, there might have been changes to the ETF data. Please go to the "
            "benchmark module and choose the tickers as the subset of data."
        )
        return None  # Stops the function and exits early
    logger.debug(
        "Adding allocation targets from the naive 1/N bond/stock allocation strategy, "
        "that starts with 100% invested in stocks and glides towards 0% invested in stocks."
    )
    years = [str(i + 2023) for i in annual_alloc_targets.index]
    asset_class = create_asset_class_df(tickers)
    # Initialize allocation dataframe with zeros
    allocation_df = pd.DataFrame(
        0, index=annual_alloc_targets.index, columns=asset_class.index
    )

    # Classify assets into risky and non_risky categories
    risky_assets = asset_class.index[asset_class["Asset Class"] == "risky"].tolist()
    non_risky_assets = asset_class.index[
        asset_class["Asset Class"] == "non_risky"
    ].tolist()

    # Calculate the number of assets in each category
    num_risky = len(risky_assets)
    num_non_risky = len(non_risky_assets)

    # Calculate allocation per asset based on annual_alloc_targets
    for year in annual_alloc_targets.index:
        allocation_df.loc[year, risky_assets] = (
            annual_alloc_targets.loc[year, 0] / num_risky if num_risky else 0
        )
        allocation_df.loc[year, non_risky_assets] = (
            (1 - annual_alloc_targets.loc[year, 0]) / num_non_risky
            if num_non_risky
            else 0
        )
    allocation_df.index = years
    return allocation_df


# asset_class_df = calculate_target_allocation(annual_alloc_targets, mst_ISIN_all)
