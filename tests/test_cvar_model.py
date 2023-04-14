import pytest
from pathlib import Path
import os
import numpy as np
import pandas as pd

from models.ScenarioGeneration import ScenarioGenerator
from models.CVaRtargets import get_cvar_targets

TEST_DIR = Path(__file__).parent

# Create RNG for testing 
test_rng = np.random.default_rng(0)
sg = ScenarioGenerator(test_rng)

# Load test data
weeklyReturns = pd.read_parquet(os.path.join(TEST_DIR, 'data/all_etfs_rets.parquet.gzip'))
names = pd.read_parquet(os.path.join(TEST_DIR, 'data/all_etfs_rets_name.parquet.gzip')).columns.values
tickers = weeklyReturns.columns.values

# Set test dates
start_test_date = pd.to_datetime("2017-07-01")
end_test_date = pd.to_datetime("2022-07-20")

# Find Benchmarks' ISIN codes
benchmarks = ['iShares MSCI All Country Asia ex Japan Index Fund ETF', 'iShares MSCI ACWI ETF']
benchmark_isin = [tickers[list(names).index(name)] for name in benchmarks]

# Set testing data
test_dataset = weeklyReturns[(weeklyReturns.index > start_test_date) & (weeklyReturns.index <= end_test_date)].copy()


def test_get_cvar_targets():
    expected_targets = pd.read_csv("tests/targets_BASE.csv", index_col=0)
    expected_benchmark_port_val = pd.read_csv("tests/benchmark_port_val_BASE.csv", index_col=0, parse_dates=True)

    start_of_test_dataset = str(test_dataset.index.date[0])
    targets, benchmark_port_val = get_cvar_targets(
        test_date=start_of_test_dataset,
        benchmark=benchmark_isin, 
        budget=100,
        cvar_alpha=0.05,
        data=weeklyReturns,
        scgen=sg
    )

    pd.testing.assert_frame_equal(targets, expected_targets)
    pd.testing.assert_frame_equal(benchmark_port_val, expected_benchmark_port_val)
