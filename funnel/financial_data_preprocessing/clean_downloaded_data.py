import os

import pandas as pd


def clean_data(data_raw):
    """
    Function to clean raw data from Yahoo! finance and transform it into weekly returns
    """
    data_raw = data_raw.fillna("")

    # Delete tickers for which we don't have data for the whole time period
    for column in data_raw.columns:
        # if the first three or last three values of the column are not empty, then delete the column
        if (
            not data_raw[column].values[:3].all()
            or not data_raw[column].values[-3:].all()
        ):
            data_raw.drop(column, axis=1, inplace=True)

    # Fill missing daily prices with the closest available price in the future
    data = data_raw.copy()
    for asset in data_raw.columns:
        for indx, date in enumerate(data_raw.index):
            if not data_raw.loc[date, asset]:
                for date_future in list(data_raw.index)[indx:]:
                    if data_raw.loc[date_future, asset]:
                        data.loc[date, asset] = data_raw.loc[date_future, asset]
                        print("found price")
                        break
                    else:
                        continue

    # Delete tickers (outliers) with daily returns bigger that 20%
    to_delete = []
    for asset in data.columns:
        column = list(data[asset])
        value_old = column[0]
        for value in column[1:]:
            if abs((value / value_old) - 1) > 0.20:
                to_delete.append(asset)
                print(asset, (value / value_old) - 1, len(to_delete))
                break
            else:
                value_old = value
    for delete_col in to_delete:
        data.drop(delete_col, axis=1, inplace=True)

    # Select only Wednesdays to be able to compute monthly returns
    data_wed = data[data.index.weekday == 2]

    # Check if we have all Wednesdays' prices, if not fill it with the price 5 days in the past
    date_test = data_wed.index[0]
    date_list = data_wed.index.to_list()
    while date_test < date_list[-1]:
        date_test = date_test + pd.Timedelta(days=7)
        if date_test not in data_wed.index:
            print(date_test)
            data_wed.loc[date_test] = data.loc[
                date_test - pd.Timedelta(days=5)
            ].to_list()

    # Sort df by index
    data_wed = data_wed.sort_index()

    # Create dataframes with returns instead of prices
    data_wed_rets = data_wed.copy()
    for asset in data_wed.columns:
        data_wed_rets[asset] = data_wed[asset].pct_change()

    # drop the first row, because it contains NaNs
    data_wed_rets = data_wed_rets.drop(data_wed_rets.index[0])

    wanted_columns = [col for col in data_wed_rets.columns if col[0] != "nan"]
    data_wed_rets = data_wed_rets[wanted_columns]
    # Save results with returns into data folder for the app
    data_wed_rets.to_parquet(
        os.path.join(
            os.path.dirname(os.getcwd()), "financial_data/all_etfs_rets.parquet.gzip"
        ),
        compression="gzip",
    )


if __name__ == "__main__":
    daily_prices = pd.read_parquet(
        os.path.join(os.path.dirname(os.getcwd()), "financial_data/daily_price.parquet")
    )

    # Select just some indices
    subset_data = daily_prices[
        (daily_prices.index > "2013-01-01") & (daily_prices.index < "2024-07-28")
    ]

    # Clean data and save for the investment funnel app
    clean_data(data_raw=subset_data)
