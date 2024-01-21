import os
from typing import List

import pandas as pd
import yfinance as yf
from loguru import logger


def download_data(start_date: str, end_date: str, tickers: List[str]) -> pd.DataFrame:
    """
    Function to download all needed ETF data for NORD and Lysa portfolios
    """

    # Download price data from Yahoo! finance based on list of ETF tickers and start/end dates
    try:
        daily_prices = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
    except Exception as e:
        logger.warning(f"⚠️ Problem when downloading our data with an error: {e}")
        daily_prices = None

    return daily_prices


if __name__ == "__main__":
    # Load tickers' names
    path_to_tickers = os.path.join(
        os.path.dirname(os.getcwd()), "financial_data/top_2000_etfs.xlsx"
    )
    data_excel = pd.read_excel(path_to_tickers)
    tickers = data_excel["List of Top 100 ETFs"].to_list()[1:]
    mapping = dict(
        zip(
            data_excel["List of Top 100 ETFs"].to_list()[1:],
            data_excel["Unnamed: 1"].to_list()[1:],
        )
    )

    # Download raw data
    data_yahoo = download_data(
        start_date="2022-12-31", end_date="2023-07-30", tickers=tickers
    )
    data_yahoo.columns = [
        data_yahoo.columns,
        [mapping[col] for col in data_yahoo.columns],
    ]
    data_yahoo.to_parquet(
        os.path.join(os.path.dirname(os.getcwd()), "financial_data/daily_price.parquet")
    )
