import pytest
from pathlib import Path
import os
import numpy as np
import pandas as pd

from models.ScenarioGeneration import ScenarioGenerator
from models.CVaRtargets import get_cvar_targets
from models.CVaRmodel import cvar_model
from models.MST import minimum_spanning_tree


TEST_DIR = Path(__file__).parent


@pytest.fixture()
def rng():
    test_rng = np.random.default_rng(0)
    return test_rng


@pytest.fixture()
def scgen(rng):
    sg = ScenarioGenerator(rng)
    return sg


@pytest.fixture(scope="module")
def weeklyReturns():
    weekly_returns = pd.read_parquet(os.path.join(TEST_DIR, 'data/all_etfs_rets.parquet.gzip'))
    return weekly_returns


@pytest.fixture(scope="module")
def tickers(weeklyReturns):
    tickers = weeklyReturns.columns.values
    return tickers


@pytest.fixture(scope="module")
def names():
    df_names = pd.read_parquet(os.path.join(TEST_DIR, 'data/all_etfs_rets_name.parquet.gzip'))
    names = df_names.columns.values
    return names


@pytest.fixture(scope="module")
def start_train_date():
    return  pd.to_datetime("2014-06-11")


@pytest.fixture(scope="module")
def end_train_date():
    return  pd.to_datetime("2017-07-01")


@pytest.fixture(scope="module")
def start_test_date():
    return  pd.to_datetime("2017-07-01")


@pytest.fixture(scope="module")
def end_test_date():
    return  pd.to_datetime("2022-07-20")


@pytest.fixture(scope="module")
def benchmark_isin(tickers, names):
    benchmarks = ['iShares MSCI All Country Asia ex Japan Index Fund ETF', 'iShares MSCI ACWI ETF']
    benchmark_isin = [tickers[list(names).index(name)] for name in benchmarks]
    return benchmark_isin


@pytest.fixture(scope="module")
def train_dataset(weeklyReturns, start_train_date, end_train_date):
    train_dataset = weeklyReturns[(weeklyReturns.index >= start_train_date) & (weeklyReturns.index <= end_train_date)].copy()
    return train_dataset


@pytest.fixture(scope="module")
def test_dataset(weeklyReturns, start_test_date, end_test_date):
    test_dataset = weeklyReturns[(weeklyReturns.index > start_test_date) & (weeklyReturns.index <= end_test_date)].copy()
    return test_dataset


@pytest.fixture(scope="module")
def subset_of_assets(train_dataset):
    n_mst_runs = 2
    subset_mst_df = train_dataset
    for i in range(n_mst_runs):
        subset_mst, subset_mst_df, corr_mst_avg, pdi_mst = minimum_spanning_tree(subset_mst_df)
    return subset_mst


@pytest.fixture(scope="module")
def n_simulations():
    return 1000


@pytest.fixture()
def scenarios(train_dataset, test_dataset, subset_of_assets, n_simulations, scgen):
    scenarios = scgen.monte_carlo(
        data=train_dataset.loc[:, train_dataset.columns.isin(subset_of_assets)],
        n_simulations=n_simulations,
        n_test=len(test_dataset.index)
    )
    return scenarios


@pytest.fixture()
def cvar_target_data(test_dataset, weeklyReturns, benchmark_isin, scgen):
    start_of_test_dataset = str(test_dataset.index.date[0])
    targets, benchmark_port_val = get_cvar_targets(
        test_date=start_of_test_dataset,
        benchmark=benchmark_isin,  # MSCI World benchmark
        budget=100,
        cvar_alpha=0.05,
        data=weeklyReturns,
        scgen=scgen
    )
    return targets, benchmark_port_val


def test_get_cvar_targets(test_dataset, benchmark_isin, weeklyReturns, scgen):
    expected_targets = pd.read_csv("tests/targets_BASE.csv", index_col=0)
    expected_benchmark_port_val = pd.read_csv("tests/benchmark_port_val_BASE.csv", index_col=0, parse_dates=True)

    start_of_test_dataset = str(test_dataset.index.date[0])
    targets, benchmark_port_val = get_cvar_targets(
        test_date=start_of_test_dataset,
        benchmark=benchmark_isin, 
        budget=100,
        cvar_alpha=0.05,
        data=weeklyReturns,
        scgen=scgen
    )

    pd.testing.assert_frame_equal(targets, expected_targets)
    pd.testing.assert_frame_equal(benchmark_port_val, expected_benchmark_port_val)


def test_cvar_model(test_dataset, subset_of_assets, scenarios, cvar_target_data):
    expected_port_allocation = pd.read_csv("tests/port_allocation_BASE.csv", index_col=0)
    expected_port_value = pd.read_csv("tests/port_value_BASE.csv", index_col=0, parse_dates=True)
    expected_port_cvar = pd.read_csv("tests/port_cvar_BASE.csv", index_col=0)

    targets, _ = cvar_target_data
    
    port_allocation, port_value, port_cvar = cvar_model(
        test_ret=test_dataset[subset_of_assets],        
        scenarios=scenarios,  # Scenarios
        targets=targets,  # Target
        budget=100,
        cvar_alpha=0.05,
        trans_cost=0.001,
        max_weight=1
    )

    #port_allocation.to_csv("tests/port_allocation_ACTUAL.csv")
    #port_value.to_csv("tests/port_value_ACTUAL.csv")
    #port_cvar.to_csv("tests/port_cvar_ACTUAL.csv")

    active_constraints = (targets.to_numpy() - port_cvar.to_numpy()) < 1e-5
    pd.testing.assert_frame_equal(port_allocation, expected_port_allocation)
    pd.testing.assert_frame_equal(port_value, expected_port_value)
    pd.testing.assert_frame_equal(port_cvar[active_constraints], expected_port_cvar[active_constraints])
