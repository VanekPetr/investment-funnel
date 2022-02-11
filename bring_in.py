from inventory import *
import yfinance as yf
import pandas as pd
import math
import pytz
import sys

sys.path.append('/investment-funnel/data')


def _yahoo_import(symbol):
    data = yf.download(symbol, interval="1d", period="max")
    daily_prices = data['Adj Close']

    return daily_prices


dailyPricesIsin = _yahoo_import(tickers)
dailyPricesName = dailyPricesIsin.rename(columns=inv_map)  # inv_map from inventory

# Start end date
start_date = "2020-10-02 00:00:00"

data_name = dailyPricesName[dailyPricesName.index >= start_date]
data_isin = dailyPricesIsin[dailyPricesIsin.index >= start_date]

data_name = data_name[data_name.index < data_name.index[-1]]
data_isin = data_isin[data_isin.index < data_isin.index[-1]]

# if first nan, delete
to_drop_isin = []
to_drop_name = []

for col1, col2 in zip(data_isin.columns, data_name.columns):
    try:
        if math.isnan(data_name[str(col2)][0]):
            to_drop_name.append(col2)
        if math.isnan(data_isin[str(col1)][0]):
            to_drop_isin.append(col1)
        if math.isnan(data_name[str(col2)][-2]):
            to_drop_name.append(col2)
        if math.isnan(data_isin[str(col1)][-2]):
            to_drop_isin.append(col1)
    except:
        to_drop_isin.append(col1)
        to_drop_name.append(col2)

data_name = data_name.drop(columns=to_drop_name, axis=1)
data_isin = data_isin.drop(columns=to_drop_isin, axis=1)

# then loop the rest
for k in range(len(data_name.columns)):
    for i in range(len(data_name.index)):
        if math.isnan(float(data_name.iloc[i, k])):
            data_name.iloc[i, k] = data_name.iloc[i - 1, k].copy()
            data_isin.iloc[i, k] = data_isin.iloc[i - 1, k].copy()

# GET WEEKLY RETURNS
# Get prices only for Wednesdays and delete Nan columns
pricesWed_name = data_name[data_name.index.weekday == 2].dropna(axis=1)
pricesWed_isin = data_isin[data_isin.index.weekday == 2].dropna(axis=1)

# Get weekly returns
weeklyReturns_name = pricesWed_name.pct_change().drop(pricesWed_name.index[0])  # drop first NaN row
weeklyReturns_isin = pricesWed_isin.pct_change().drop(pricesWed_isin.index[0])  # drop first NaN row

# Name - From parquet to df and merge into one unique name df
temp_name_df = pd.read_parquet('data/algostrata_name.parquet', engine='pyarrow')
weeklyReturns_name = weeklyReturns_name.tz_localize(pytz.utc).tz_convert(tz='UTC')
temp_name_df = temp_name_df.join(weeklyReturns_name)

# Isin - From parquet to df and merge into one unique isin df
temp_isin_df = pd.read_parquet('data/algostrata_isin.parquet', engine='pyarrow')
weeklyReturns_isin = weeklyReturns_isin.tz_localize(pytz.utc).tz_convert(tz='UTC')
temp_isin_df = temp_isin_df.join(weeklyReturns_isin)

# Substitute NaN values with 0
temp_name_df = temp_name_df.fillna(0)
temp_isin_df = temp_isin_df.fillna(0)

# To parquet, overwrites the existing parquet files
temp_name_df.to_parquet('data/algostrata_name.parquet')
temp_isin_df.to_parquet('data/algostrata_isin.parquet')
