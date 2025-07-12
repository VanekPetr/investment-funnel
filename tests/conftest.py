from pathlib import Path

import pytest
from ifunnel.models.main import initialize_bot

@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "resources"

@pytest.fixture(scope="session")
def algo(resource_dir: Path):
    """
    Initialize the investment bot for testing.

    This fixture initializes the investment bot with default parameters
    and makes it available for tests.

    Returns:
        The initialized investment bot
    """
    return initialize_bot(file = resource_dir / "all_etfs_rets.parquet.gzip")
