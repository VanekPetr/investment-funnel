from loguru import logger
import yfinance as yf
import datetime

# *** Prepare our data ***
# Dictionary with ETF's full name and its ticker
lysa_stock_etfs = {
    'Vanguard US': 'VUN.TO',
    'Vanguard North America': 'VNRT',   # Nan
    'Vanguard Europe': 'VGK',
    'Vanguard Emerging Markets': 'VWO',
    'Vanguard Global Small-Cap': 'VB',
    'iShares US': 'ITOT',
    'Vanguard Japan': 'VJPN',   # Nan
    'Vanguard Pacific': 'VPL',
    'Lyxor Europe': 'DR'    # Nan
}

lysa_bond_etfs = {
    'Vanguard Europe Government Bond': 'VETY',  # Nan
    'iShares Europe Government Bond': 'LU',
    'Vanguard Global Bond': 'BNDW',
    'Vanguard Global Short-Term Bond': 'BSV',
    'Vanguard Euro Investment Grade Bond': 'VECP',
    'Vanguard Eurozone Inflation-Linked Bond': 'VTIP',
    'Vanguard Global Aggregate Bond': 'VGAB',   # Nan
    'iShares Global Inflation Linked Govt Bond': 'IGIL',    # Nan
    'Vanguard EUR Corporate Bond': 'VECP',  # Nan
    'iShares Core € Corp Bond': 'IEAC'  # Nan
}

nord_etfs = {
    '... the rest if for Edin :)': 'ticker name which can be found for example on etf.com'
}

# *** Prepare data for yFinance ***
all_tickers = []
# Get list of all tickers
for dictionary in [lysa_stock_etfs, lysa_bond_etfs]:    # add nord_etfs here when ready
    all_tickers.extend(dictionary.values())

# *** Download the data ***
# Select start and end date for you data
start_date = '2020-11-20'
end_date = '2022-01-20'

# Download price data from Yahoo! finance based on list of ETF tickers and start/end dates
try:
    daily_prices = yf.download(all_tickers, start=start_date, end=end_date)['Adj Close']
except Exception as e:
    logger.warning(f"Problem when downloading our data with an error: {e}")

