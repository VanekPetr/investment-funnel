import os

import gams.transfer as gt
import pandas as pd
from tqdm import tqdm


def save_into_gdx(monthly_returns_df):
    # Initialize GAMS connection
    working_dir = os.getcwd()

    # *** CREATE GDX FILE ***
    # Initialization for GAMS version v47
    # Documentation: https://www.gams.com/latest/docs/API_PY_GAMSTRANSFER.html#PY_GAMSTRANSFER_INSTALLATION
    m = gt.Container(
        system_directory="/Library/Frameworks/GAMS.framework/Versions/47/Resources/"
    )

    # Sets
    asset_set = m.addSet(
        "Asset",
        records=[a[0] for a in monthly_returns_df.columns],
        description="Asset ISIN",
    )
    asset_name_set = m.addSet(
        "AssetName",
        records=[a[1] for a in monthly_returns_df.columns],
        description="Asset Name",
    )
    date_set = m.addSet(
        "Date",
        records=[str(date.date()) for date in monthly_returns_df.index],
        description="Date",
    )

    # Parameter
    # Save weekly returns in a gams needed format
    returns_gams = pd.DataFrame(columns=["Date", "Asset", "AssetName", "Return"])
    for date in tqdm(monthly_returns_df.index):
        for asset in monthly_returns_df.columns:
            return_value = monthly_returns_df.loc[date, asset]
            returns_gams.loc[len(returns_gams)] = [
                str(date.date()),
                asset[0],
                asset[1],
                float(return_value),
            ]

    m.addParameter(
        "AssetReturn",
        domain=[date_set, asset_set, asset_name_set],
        records=returns_gams,
        description="Weekly adjusted returns",
    )

    m.write(os.path.join(os.path.dirname(working_dir), "financial_data/input_data.gdx"))


if __name__ == "__main__":
    monthly_returns = pd.read_parquet(
        os.path.join(
            os.path.dirname(os.getcwd()),
            "financial_data/all_etfs_rets_2024.parquet.gzip",
        )
    )

    save_into_gdx(monthly_returns)
