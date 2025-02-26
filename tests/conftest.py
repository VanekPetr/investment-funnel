from pathlib import Path

import numpy as np
import pytest
from ifunnel.models.main import initialize_bot


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "resources"

@pytest.fixture(scope="session", name="algo")
def algo_fixture(resource_dir):
    return initialize_bot(resource_dir / "all_etfs_rets.parquet.gzip")


@pytest.fixture()
def rng():
    test_rng = np.random.default_rng(seed=42)
    return test_rng
