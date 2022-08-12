import gamstransfer as gt
import pandas as pd
import os
from data_preparation.yahoo_download import download_data

# Initialize GAMS connection
working_dir = os.getcwd()

# Load tickers' names
path = 'data/top_2000_etfs.xlsx'
data_excel = pd.read_excel(path)
tickers = data_excel['List of Top 100 ETFs'].to_list()[1:]

# Download raw data
data_raw = download_data(start_date='2014-05-30', end_date='2022-07-30', tickers=tickers)
data_raw = data_raw.fillna('')

# Delete tickers fo which we don't have data for the whole time period
for column in data_raw.columns:
    # if the first three or last three values of the column are not empty, then delete the column
    if not data_raw[column].values[:3].all() or not data_raw[column].values[-3:].all():
        data_raw.drop(column, axis=1, inplace=True)

# Fill missing daily prices with the closest available price in the future
data = data_raw.copy()
for asset in data_raw.columns:
    for indx, date in enumerate(data_raw.index):
        if not data_raw.loc[date, asset]:
            for date_future in list(data_raw.index)[indx:]:
                if data_raw.loc[date_future, asset]:
                    data.loc[date, asset] = data_raw.loc[date_future, asset]
                    print('found price')
                    break
                else:
                    continue

# Delete tickers (outliers) with daily returns bigger that 15%
to_delete = []
for asset in data.columns:
    column = list(data[asset])
    value_old = column[0]
    for value in column[1:]:
        if abs((value/value_old)-1) > 0.15:
            to_delete.append(asset)
            print(asset, (value/value_old)-1, len(to_delete))
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
        data_wed.loc[date_test] = data.loc[date_test - pd.Timedelta(days=5)].to_list()

# Sort df by index
data_wed = data_wed.sort_index()

# *** CREATE GDX FILE ***
# Initialization
m = gt.Container(system_directory='/Library/Frameworks/GAMS.framework/Versions/39/Resources/')

# Sets
Asset = m.addSet('Asset', records=list(data_wed.columns), description='All Assets (ETFs)')
Date = m.addSet('Date', records=list(data_wed.index)[1:], description='Date')

# Parameter
# Compute weekly returns in a gams needed format
returns_gams = pd.DataFrame(columns=['Date', 'Asset', 'Return'])
previous_date = data_wed.index[0]
counter = 0
for date in list(data_wed.index)[1:]:
    for asset in data_wed.columns:
        price_yesterday = data_wed.loc[previous_date, asset]
        price_today = data_wed.loc[date, asset]
        return_value = (price_today/price_yesterday)-1
        if abs(return_value) > 0.1:
            counter += 1
            print(return_value, date, asset)
        returns_gams.loc[len(returns_gams)] = [str(date.date()), asset, return_value]
    previous_date = date

print(counter)
ETFreturns = m.addParameter('AssetReturns', domain=[Date, Asset], records=returns_gams,
                            description='Weekly adjusted returns')


m.write(os.path.join(working_dir, 'data/input_data.gdx'))
returns_gams.to_parquet('data/all_etfs_rets_to_clean.parquet.gzip', compression='gzip')
data_wed.to_parquet('data/all_etfs_prices.parquet.gzip', compression='gzip')

# New dataframe with names instead of tickers
name_mapping = dict(zip(data_excel['List of Top 100 ETFs'].to_list()[1:], data_excel['Unnamed: 1'].to_list()[1:]))
data_wed_name = data_wed.copy()
new_column = [name_mapping[asset] for asset in data_wed.columns]
data_wed_name.columns = new_column
data_wed_name.to_parquet('data/all_etfs_prices_name.parquet.gzip', compression='gzip')

# Create dataframes with returns instead of prices
data_wed_rets = data_wed.copy()
for asset in data_wed.columns:
    data_wed_rets[asset] = data_wed[asset].pct_change()
# drop the first row, because it contains NaNs
data_wed_rets = data_wed_rets.drop(data_wed_rets.index[0])
data_wed_rets.to_parquet('data/all_etfs_rets.parquet.gzip', compression='gzip')
data_wed_rets_name = data_wed_rets.copy()
new_column = [name_mapping[asset] for asset in data_wed_rets.columns]
data_wed_rets_name.columns = new_column
data_wed_rets_name.to_parquet('data/all_etfs_rets_name.parquet.gzip', compression='gzip')
