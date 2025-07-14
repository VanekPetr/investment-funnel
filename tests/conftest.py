from pathlib import Path
from typing import Any

import pytest
from ifunnel.models.main import initialize_bot


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture() -> Path:
    """
    Provide the path to the test resources directory.

    This fixture returns the absolute path to the resources directory
    containing test data files needed for the tests.

    Returns:
        Path: The absolute path to the resources directory
    """
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="session")
def algo(resource_dir: Path) -> Any:
    """
    Initialize the investment bot for testing.

    This fixture initializes the investment bot with default parameters
    and makes it available for tests. It uses a test dataset from the
    resources directory.

    Args:
        resource_dir (Path): The path to the resources directory

    Returns:
        Any: The initialized investment bot instance
    """
    return initialize_bot(file = resource_dir / "all_etfs_rets.parquet.gzip")
