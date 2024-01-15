from datetime import timedelta

import pandas as pd
import pytest

from funnel.models.MVOmodel import mvo_model
from funnel.models.MVOtargets import get_mvo_targets


@pytest.fixture(scope="module")
def mvo_target_data(start_test_date, weekly_returns, request):
    start_of_test_dataset = str(start_test_date + timedelta(days=1))
    targets, benchmark_port_val = get_mvo_targets(
        test_date=start_of_test_dataset,
        benchmark=request.getfixturevalue(request.param),  # MSCI World benchmark
        budget=100,
        data=weekly_returns,
    )
    return targets, benchmark_port_val


@pytest.mark.parametrize(
    "mvo_target_data, label",
    [("benchmark_isin_1", "1"), ("benchmark_isin_2", "2")],
    indirect=["mvo_target_data"],
)
def test_get_mvo_targets(mvo_target_data, label, resource_dir):
    expected_targets = pd.read_csv(
        resource_dir / f"mvo/targets_{label}_BASE.csv", index_col=0
    )
    expected_benchmark_port_val = pd.read_csv(
        resource_dir / f"mvo/benchmark_port_val_{label}_BASE.csv",
        index_col=0,
        parse_dates=True,
    )
    expected_benchmark_port_val.index = expected_benchmark_port_val.index.astype(
        "datetime64[us]"
    )

    targets, benchmark_port_val = mvo_target_data

    # targets.to_csv(f"tests/mvo/targets_{label}_ACTUAL.csv")
    # benchmark_port_val.to_csv(f"tests/mvo/benchmark_port_val_{label}_ACTUAL.csv")
    pd.testing.assert_frame_equal(targets, expected_targets)
    pd.testing.assert_frame_equal(benchmark_port_val, expected_benchmark_port_val)


@pytest.mark.parametrize("mvo_target_data", ["benchmark_isin_2"], indirect=True)
def test_mvo_model(test_narrow_dataset, moments, mvo_target_data, resource_dir):
    expected_port_allocation = pd.read_csv(
        resource_dir / "mvo/port_allocation_BASE.csv", index_col=0
    )
    expected_port_value = pd.read_csv(
        resource_dir / "mvo/port_value_BASE.csv", index_col=0, parse_dates=True
    )
    expected_port_risk = pd.read_csv(
        resource_dir / "mvo/port_risk_BASE.csv", index_col=0
    )

    targets, _ = mvo_target_data
    sigma_lst, mu_lst = moments

    port_allocation, port_value, port_risk = mvo_model(
        mu_lst=mu_lst,
        sigma_lst=sigma_lst,
        test_ret=test_narrow_dataset,
        targets=targets,  # Target
        budget=100,
        trans_cost=0.001,
        max_weight=1,
        solver="ECOS",
        lower_bound=0,
    )

    # port_allocation.to_csv("mvo/port_allocation_BASE.csv")
    # port_value.to_csv("mvo/port_value_BASE.csv")
    # port_risk.to_csv("mvo/port_risk_BASE.csv")

    active_constraints = (targets.to_numpy() - port_risk.to_numpy()) < 1e-5
    pd.testing.assert_frame_equal(port_allocation, expected_port_allocation, atol=1e-5)
    pd.testing.assert_frame_equal(port_value, expected_port_value)
    pd.testing.assert_frame_equal(
        port_risk[active_constraints], expected_port_risk[active_constraints]
    )
