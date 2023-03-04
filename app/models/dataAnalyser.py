import pandas as pd
import numpy as np


# Function computing the geometric mean of annual returns
def mean_an_returns(data):
    result = 1

    for i in range(len(data.index)):
        result *= (1 + data.iloc[i, :])

    result = result ** (1 / float(len(data.index) / 52)) - 1

    return result


# Function computing the final statistics of the backtesting
def final_stats(data):
    # TABLE WITH AVG RET AND STD OF RET
    data = data.pct_change()
    data = data.drop(data.index[:1])

    mu_ga = round(mean_an_returns(data), 2)  # annual geometric mean
    std_dev_a = round(data.std(axis=0) * np.sqrt(52), 2)  # standard deviation of Annual Returns

    stat_df = pd.concat([mu_ga, std_dev_a], axis=1)  # table
    stat_names = ["Average Annual Returns", "Standard Deviation of Returns"]
    stat_df.columns = stat_names  # add names

    # COMPUTE SHARPE RATIO AND ADD IT INTO THE TABLE
    sharpe = round(stat_df.loc[:, "Average Annual Returns"] / stat_df.loc[:, "Standard Deviation of Returns"], 2)
    stat_df = pd.concat([stat_df, sharpe], axis=1)  # add sharpe ratio into the table
    stat_names = ["Avg An Ret", "Std Dev of Ret", "Sharpe R"]
    stat_df.columns = stat_names

    return stat_df


# Function returning weekly returns
def get_weekly_returns(data):
    # DEFINE IF WE WORK WITH ISIN CODES OR NAMES OF MUTUAL FUNDS
    prices_df = data
    # MODIFY THE DATA
    prices_on_wed = prices_df[prices_df.index.weekday == 2]  # Only wednesdays

    # Get weekly returns
    weekly_returns = prices_on_wed.pct_change()
    weekly_returns = weekly_returns.drop(weekly_returns.index[:1])  # drop first NaN row

    return weekly_returns


tickers = ['AAXJ', 'ACWI', 'AGG', 'AMLP', 'AOA', 'AOK', 'AOR', 'BIL', 'BKLN', 'BLV', 'BND', 'BSV', 'CORN', 'CQQQ',
           'DBA', 'DBO', 'DIA', 'DJP', 'DUST', 'DVY', 'EEM', 'EFA', 'EMB', 'ERX', 'EWG', 'EWH', 'EWJ', 'EWL', 'EWN',
           'EWT', 'EWW', 'EWY', 'EWZ', 'EZU', 'FAS', 'FAZ', 'FEZ', 'FXI', 'GDX', 'GDXJ', 'GLD', 'HDV', 'HYG', 'IAEX.L',
           'IAU', 'ABB', 'IDEM.L', 'IEF', 'IEML.L', 'IFFF.L', 'IHI', 'IJH', 'IJJ', 'IJPE.L', 'IJR', 'ILTB', 'IMEU.L',
           'IMIB.L', 'ISF.L', 'ITOT', 'ITWN.L', 'IUKP.L', 'IUSA.L', 'IUSG', 'IUSV', 'IVV', 'IVW', 'IWB', 'IWF', 'IWM',
           'IWN', 'IWO', 'IWR', 'IWS', 'IXJ', 'JNK', 'KBE', 'KRE', 'LIT', 'LQD', 'MCHI', 'MDY', 'MINT', 'MUB', 'NUGT',
           'OIH', 'PALL', 'PFF', 'PGX', 'PHYS', 'PPLT', 'PSLV', 'QQQ', 'RSX', 'RWR', 'SCHE', 'SCHF', 'SCHX', 'SCO',
           'SDIV', 'SDOW', 'SDS', 'SEMB.L', 'SH', 'SHV', 'SHY', 'SKYY', 'SLV', 'SMH', 'SOXL', 'SOXS', 'SOXX', 'SPLV',
           'SPXL', 'SPXS', 'SPXU', 'SPY', 'SPYG', 'SQQQ', 'SRTY', 'SSO', 'SWDA.L', 'TAN', 'TFI', 'THD', 'TIP', 'TLT',
           'TMF', 'TNA', 'TQQQ', 'TVIX', 'TZA', 'UCO', 'UDOW', 'UGA', 'UNG', 'UPRO', 'USL', 'USO', 'USRT', 'VB', 'VBK',
           'VBR', 'VCIT', 'VCSH', 'VEA', 'VEU', 'VFH', 'VGK', 'VGT', 'VHT', 'VIG', 'VNQ', 'VO', 'VOE', 'VONG', 'VOO',
           'VOOG', 'VOOV', 'VOX', 'VTI', 'VTV', 'VUG', 'VWO', 'VXUS', 'XBI', 'XCX5.L', 'XLB', 'XLE', 'XLF', 'XLI',
           'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XOP', 'XS6R.L', 'YINN']
