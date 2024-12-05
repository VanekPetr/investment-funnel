import os

import dateutil.parser
import numpy as np
import pandas as pd
import requests

from ..settings import settings


# BATCH FUNCTION
# ----------------------------------------------------------------------
def batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


# Get IDs and ISIN codes
# ----------------------------------------------------------------------
def get_algostrata_data() -> pd.DataFrame:
    idList = []  # empty list of IDs
    isinList = []  # empty list of isin codes
    nameList = []  # empty list of names
    # GET ASSET NAME DATA
    response = requests.get(
        settings.ALGOSTRATA_NAMES_URL,
        headers={
            "X-Api-Key": settings.ALGOSTRATA_KEY,
            "content-type": "application/json",
        },
    )
    data = response.json()  # downloaded data
    # SAVE IDs and ISIN CODES INTO LISTS
    for asset in data:
        idList.append(asset["id"])
        isinList.append(asset["isin"])
        nameList.append(asset["name"])

    # Get the price data with index
    # ----------------------------------------------------------------------
    batchSize = 3  # size of a batch
    roundRep = int(np.ceil(len(idList) / batchSize))  # number of iterations
    rep = 0  # current iteration
    firstRun = True
    # LOAD DATASET BY STEP, EACH STEP XY ASSETS
    for subIdList in batch(idList, batchSize):
        # GET ASSET PRICE DATA
        print("---- Starting round", rep + 1, "out of", roundRep, "----")
        response = requests.post(
            settings.ALGOSTRATA_PRICES_URL,
            json={"idList": subIdList},
            headers={
                "X-Api-Key": settings.ALGOSTRATA_KEY,
                "content-type": "application/json",
            },
        )

        if response.status_code != 200:
            print(f"Code {response.reason}, content {response.text}")
            print("---- Error round", rep + 1, "out of", roundRep, "----")
            continue

        data = response.json()  # downloaded data

        # CREATE PANDAS TABLE WITH ALL PRICE DATA
        for num, asset in enumerate(data["result"]):
            # IF WE HAVE A PRICE DATA THEN
            if asset["priceData"] is not None:
                priceData = asset["priceData"]
                reInvestedPrices = priceData["reInvestedPrices"]
                dates = list(
                    map(lambda x: dateutil.parser.parse(x["date"]), reInvestedPrices)
                )
                prices = list(map(lambda x: x["unit_DKK"], reInvestedPrices))

                # IF THE FIRST RUN, THEN CREATE A TABLE
                if firstRun:
                    daily_prices = pd.DataFrame(
                        prices, index=dates, columns=[isinList[0:1], nameList[0:1]]
                    )
                    firstRun = False
                # IF NOT THE FIRST RUN, JUST CONCAT THE COLUMN INTO EXISTING TABLE
                else:
                    df = pd.DataFrame(
                        prices,
                        index=dates,
                        columns=[
                            isinList[rep * batchSize + num : rep * batchSize + num + 1],
                            nameList[rep * batchSize + num : rep * batchSize + num + 1],
                        ],
                    )
                    # IF THE PRICE DATA ARE NOT ALL NaN, THEN
                    if not df.isnull().values.all():
                        daily_prices = pd.concat([daily_prices, df], axis=1)
        rep += 1

    return daily_prices


if __name__ == "__main__":
    # Download raw data
    data_algostrata = get_algostrata_data()
    # Save daily_prices into parquet file
    data_algostrata.to_parquet(
        os.path.join(os.path.dirname(os.getcwd()), "financial_data/daily_price.parquet")
    )
