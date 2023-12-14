import numpy as np
import pandas as pd


def mean_an_returns(data: pd.DataFrame) -> float:
    """Function computing the geometric mean of annual returns"""

    result = 1
    for i in range(len(data.index)):
        result *= 1 + data.iloc[i, :]

    result = result ** (1 / float(len(data.index) / 52)) - 1

    return result


def final_stats(data: pd.DataFrame) -> pd.DataFrame:
    """Function computing the final statistics of the backtesting"""

    # TABLE WITH AVG RET AND STD OF RET
    data = data.pct_change()
    data = data.drop(data.index[:1])

    mu_ga = round(mean_an_returns(data), 2)  # annual geometric mean
    std_dev_a = round(
        data.std(axis=0) * np.sqrt(52), 2
    )  # standard deviation of Annual Returns

    stat_df = pd.concat([mu_ga, std_dev_a], axis=1)  # table
    stat_names = ["Average Annual Returns", "Standard Deviation of Returns"]
    stat_df.columns = stat_names  # add names

    # COMPUTE SHARPE RATIO AND ADD IT INTO THE TABLE
    sharpe = round(
        stat_df.loc[:, "Average Annual Returns"]
        / stat_df.loc[:, "Standard Deviation of Returns"],
        2,
    )
    stat_df = pd.concat([stat_df, sharpe], axis=1)  # add sharpe ratio into the table
    stat_names = ["Avg An Ret", "Std Dev of Ret", "Sharpe R"]
    stat_df.columns = stat_names

    return stat_df


def get_weekly_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Function returning weekly returns"""

    # DEFINE IF WE WORK WITH ISIN CODES OR NAMES OF MUTUAL FUNDS
    prices_df = data
    # MODIFY THE DATA
    prices_on_wed = prices_df[prices_df.index.weekday == 2]  # Only wednesdays

    # Get weekly returns
    weekly_returns = prices_on_wed.pct_change()
    weekly_returns = weekly_returns.drop(weekly_returns.index[:1])  # drop first NaN row

    return weekly_returns
