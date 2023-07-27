import pytest
from pathlib import Path
import os
import numpy as np
import pandas as pd

from models.ScenarioGeneration import ScenarioGenerator
from models.MVOtargets import get_mvo_targets
from models.MVOmodel import mvo_model
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
def whole_dataset(weeklyReturns, start_train_date, end_test_date):
    whole_dataset = weeklyReturns[(weeklyReturns.index >= start_train_date) & (weeklyReturns.index <= end_test_date)].copy()
    return whole_dataset


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


@pytest.fixture()
def mvo_inputs(whole_dataset, test_dataset, subset_of_assets, scgen):
    sigma_lst, mu_lst = ScenarioGenerator.generate_sigma_mu_for_test_periods(
        data=whole_dataset[subset_of_assets],
        n_test=len(test_dataset.index)
    )

    return sigma_lst, mu_lst


@pytest.fixture()
def mvo_target_data(test_dataset, weeklyReturns, benchmark_isin, scgen):
    start_of_test_dataset = str(test_dataset.index.date[0])
    targets, benchmark_port_val = get_mvo_targets(
        test_date=start_of_test_dataset,
        benchmark=benchmark_isin,  # MSCI World benchmark
        budget=100,
        data=weeklyReturns,
        scgen=scgen
    )
    return targets, benchmark_port_val


def test_mvo_inputs(mvo_inputs):
    expected_sigmas = np.load("tests/mvo/sigma_list_BASE.npz")
    expected_sigma_list = list((expected_sigmas[k] for k in expected_sigmas))
    expected_mus = np.load("tests/mvo/mu_list_BASE.npz")
    expected_mu_list = list((expected_mus[k] for k in expected_mus))
    
    sigma_list, mu_list = mvo_inputs
    #np.savez_compressed("tests/mvo/mu_list_ACTUAL.npz", **dict(zip([f"mu_{i}" for i in range(len(mu_list))], mu_list)))
    #np.savez_compressed("tests/mvo/sigma_list_ACTUAL.npz", **dict(zip([f"sigma_{i}" for i in range(len(sigma_list))], sigma_list)))
    np.testing.assert_array_equal(mu_list, expected_mu_list)
    np.testing.assert_array_equal(sigma_list, expected_sigma_list)


def test_get_mvo_targets(mvo_target_data):
    expected_targets = pd.read_csv("tests/mvo/targets_BASE.csv", index_col=0)
    expected_benchmark_port_val = pd.read_csv("tests/mvo/benchmark_port_val_BASE.csv", index_col=0, parse_dates=True)

    targets, benchmark_port_val = mvo_target_data

    #targets.to_csv("tests/mvo/targets_ACTUAL.csv")
    #benchmark_port_val.to_csv("tests/mvo/benchmark_port_val_ACTUAL.csv")
    pd.testing.assert_frame_equal(targets, expected_targets)
    pd.testing.assert_frame_equal(benchmark_port_val, expected_benchmark_port_val)


def test_mvo_model(test_dataset, subset_of_assets, mvo_inputs, mvo_target_data):
    expected_port_allocation = pd.read_csv("tests/mvo/port_allocation_BASE.csv", index_col=0)
    expected_port_value = pd.read_csv("tests/mvo/port_value_BASE.csv", index_col=0, parse_dates=True)
    expected_port_cvar = pd.read_csv("tests/mvo/port_cvar_BASE.csv", index_col=0)

    targets, _ = mvo_target_data
    sigma_lst, mu_lst = mvo_inputs
    
    port_allocation, port_value, port_cvar = mvo_model(
        mu_lst=mu_lst,
        sigma_lst=sigma_lst,
        test_ret=test_dataset[subset_of_assets],        
        targets=targets,  # Target
        budget=100,
        trans_cost=0.001,
        max_weight=1,
        solver="ECOS"
    )

    #port_allocation.to_csv("tests/mvo/port_allocation_ACTUAL.csv")
    #port_value.to_csv("tests/mvo/port_value_ACTUAL.csv")
    #port_cvar.to_csv("tests/mvo/port_cvar_ACTUAL.csv")

    active_constraints = (targets.to_numpy() - port_cvar.to_numpy()) < 1e-5
    pd.testing.assert_frame_equal(port_allocation, expected_port_allocation, atol=1e-5)
    pd.testing.assert_frame_equal(port_value, expected_port_value)
    pd.testing.assert_frame_equal(port_cvar[active_constraints], expected_port_cvar[active_constraints])
