import yfinance as yf
from loguru import logger


def download_data(start_date, end_date, tickers):
    """
    Function to download all needed ETF data for NORD and Lysa portfolios
    """

    # Download price data from Yahoo! finance based on list of ETF tickers and start/end dates
    try:
        daily_prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    except Exception as e:
        logger.warning(f"Problem when downloading our data with an error: {e}")
        daily_prices = None

    return daily_prices


if __name__ == "__main__":
    daily_prices_df = download_data(start_date='2020-11-20', end_date='2022-01-20')
    print(daily_prices_df)
