from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from funnel.models.MST import minimum_spanning_tree
from funnel.models.ScenarioGeneration import MomentGenerator, ScenarioGenerator


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "resources"


@pytest.fixture()
def rng():
    test_rng = np.random.default_rng(seed=42)
    return test_rng


@pytest.fixture()
def scgen(rng):
    sg = ScenarioGenerator(rng)
    return sg


@pytest.fixture(scope="session")
def weekly_returns(resource_dir):
    weekly_returns = pd.read_parquet(resource_dir / "data/all_etfs_rets.parquet.gzip")
    return weekly_returns


@pytest.fixture(scope="session")
def tickers(weekly_returns):
    tickers = weekly_returns.columns.values
    return tickers


@pytest.fixture(scope="session")
def names(resource_dir):
    df_names = pd.read_parquet(resource_dir / "data/all_etfs_rets_name.parquet.gzip")
    names = df_names.columns.values
    return names


@pytest.fixture(scope="session")
def start_train_date():
    return pd.to_datetime("2014-06-11")


@pytest.fixture(scope="session")
def end_train_date():
    return pd.to_datetime("2017-07-01")


@pytest.fixture(scope="session")
def start_test_date():
    return pd.to_datetime("2017-07-01")


@pytest.fixture(scope="session")
def end_test_date():
    return pd.to_datetime("2022-07-20")


@pytest.fixture(scope="session")
def benchmark_isin_1(tickers, names):
    benchmarks = ["iShares MSCI ACWI ETF"]
    benchmark_isin = [tickers[list(names).index(name)] for name in benchmarks]
    return benchmark_isin


@pytest.fixture(scope="session")
def benchmark_isin_2(tickers, names):
    benchmarks = [
        "iShares MSCI All Country Asia ex Japan Index Fund ETF",
        "iShares MSCI ACWI ETF",
    ]
    benchmark_isin = [tickers[list(names).index(name)] for name in benchmarks]
    return benchmark_isin


@pytest.fixture(scope="session")
def whole_wide_dataset(weekly_returns, start_train_date, end_test_date):
    whole_dataset = weekly_returns[
        (weekly_returns.index >= start_train_date)
        & (weekly_returns.index <= end_test_date)
    ].copy()
    return whole_dataset


@pytest.fixture(scope="session")
def train_wide_dataset(weekly_returns, start_train_date, end_train_date):
    train_dataset = weekly_returns[
        (weekly_returns.index >= start_train_date)
        & (weekly_returns.index <= end_train_date)
    ].copy()
    return train_dataset


@pytest.fixture(scope="session")
def subset_of_assets(train_wide_dataset):
    n_mst_runs = 2
    subset_mst_df = train_wide_dataset
    for i in range(n_mst_runs):
        subset_mst, subset_mst_df, _, _ = minimum_spanning_tree(subset_mst_df)
    return subset_mst


@pytest.fixture(scope="session")
def whole_narrow_dataset(whole_wide_dataset, subset_of_assets):
    whole_dataset = whole_wide_dataset[subset_of_assets]
    return whole_dataset


@pytest.fixture(scope="session")
def test_narrow_dataset(
    weekly_returns, start_test_date, end_test_date, subset_of_assets
):
    test_dataset = weekly_returns[
        (weekly_returns.index > start_test_date)
        & (weekly_returns.index <= end_test_date)
    ].copy()
    test_dataset = test_dataset[subset_of_assets]
    return test_dataset


@pytest.fixture(scope="session")
def length_test_dataset(test_narrow_dataset):
    return test_narrow_dataset.shape[0]


@pytest.fixture(scope="session")
def n_simulations():
    return 250


@pytest.fixture()
def mc_scenarios(
    moments, whole_narrow_dataset, length_test_dataset, n_simulations, scgen
):
    sigma_lst, mu_lst = moments

    scenarios = scgen.monte_carlo(
        data=whole_narrow_dataset,
        n_simulations=n_simulations,
        n_test=length_test_dataset,
        sigma_lst=sigma_lst,
        mu_lst=mu_lst,
    )

    return scenarios


@pytest.fixture(scope="session")
def moments(whole_narrow_dataset, length_test_dataset):
    sigma_lst, mu_lst = MomentGenerator.generate_sigma_mu_for_test_periods(
        data=whole_narrow_dataset, n_test=length_test_dataset
    )

    return sigma_lst, mu_lst
