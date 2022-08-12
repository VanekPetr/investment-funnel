import gamstransfer as gt
import pandas as pd
import os
from data_preparation.yahoo_download import download_data
from data_preparation.yahoo_data_clean import clean_data


def save_into_gdx(monthly_returns_df):

    # Initialize GAMS connection
    working_dir = os.getcwd()

    # *** CREATE GDX FILE ***
    # Initialization
    m = gt.Container(system_directory='/Library/Frameworks/GAMS.framework/Versions/39/Resources/')

    # Sets
    asset = m.addSet('Asset', records=list(monthly_returns_df.columns), description='All Assets (ETFs)')
    date = m.addSet('Date', records=list(monthly_returns_df.index)[1:], description='Date')

    # Parameter
    # Compute weekly returns in a gams needed format
    returns_gams = pd.DataFrame(columns=['Date', 'Asset', 'Return'])
    previous_date = monthly_returns_df.index[0]
    counter = 0
    for date in list(monthly_returns_df.index)[1:]:
        for asset in monthly_returns_df.columns:
            price_yesterday = monthly_returns_df.loc[previous_date, asset]
            price_today = monthly_returns_df.loc[date, asset]
            return_value = (price_today/price_yesterday)-1
            if abs(return_value) > 0.1:
                counter += 1
                print(return_value, date, asset)
            returns_gams.loc[len(returns_gams)] = [str(date.date()), asset, return_value]
        previous_date = date

    print(counter)
    etf_returns = m.addParameter('AssetReturns', domain=[date, asset], records=returns_gams,
                                 description='Weekly adjusted returns')

    m.write(os.path.join(working_dir, 'data/input_data.gdx'))


if __name__ == '__main__':
    # Load tickers' names
    path_to_tickers = 'data/top_2000_etfs.xlsx'
    data_excel = pd.read_excel(path_to_tickers)
    tickers = data_excel['List of Top 100 ETFs'].to_list()[1:]
    mapping = dict(zip(data_excel['List of Top 100 ETFs'].to_list()[1:], data_excel['Unnamed: 1'].to_list()[1:]))

    # Download raw data
    data_yahoo = download_data(start_date='2014-05-30', end_date='2022-07-30', tickers=tickers)
    # Clean data and save for the investment funnel app
    monthly_returns = clean_data(data_raw=data_yahoo, ticker_to_name_mapping=mapping, into_gdx=True)
    save_into_gdx(monthly_returns)
