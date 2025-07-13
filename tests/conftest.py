from pathlib import Path

import dash
import pytest
from ifunnel.models.main import initialize_bot

from funnel.app import load_page
from funnel.dashboard.app_callbacks import get_callbacks


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


# Fixture to create a Dash app for testing
@pytest.fixture
def app(algo) -> dash.Dash:
    """
    Create a Dash application instance for testing.

    This fixture sets up a minimal Dash application with the necessary
    components and callbacks for testing. It initializes the app with
    a basic layout and registers all callbacks.

    Args:
        algo: The algorithm bot instance

    Returns:
        Dash: Configured Dash application instance
    """
    # Create a new Dash app instance
    app = dash.Dash(__name__)

    # Apply the layout
    layout = load_page()
    app.layout = layout

    # Register all callbacks
    get_callbacks(app, algo=algo)

    return app

@pytest.fixture
def callbacks(app, algo):
    return get_callbacks(app, algo=algo)
