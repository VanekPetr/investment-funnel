from pathlib import Path
from typing import Any

import pytest
from dash import Dash, dcc, html
from ifunnel.models.main import initialize_bot

from funnel.dashboard.app_callbacks import get_callbacks
from funnel.dashboard.app_layouts import divs as app_layouts


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture() -> Path:
    """
    Provide the path to the test resources directory.

    This fixture returns a Path object pointing to the resources directory,
    which contains test data files needed for the tests.

    Returns:
        Path: Path to the resources directory
    """
    return Path(__file__).parent / "resources"


@pytest.fixture(scope="session", name="algo")
def algo_fixture(resource_dir: Path) -> Any:
    """
    Initialize the bot algorithm with test data.

    This fixture creates an instance of the investment algorithm bot
    using the test data file from the resources directory.

    Args:
        resource_dir: Path to the resources directory

    Returns:
        Any: Initialized algorithm bot instance
    """
    # Load test data from the resources directory
    return initialize_bot(resource_dir / "all_etfs_rets.parquet.gzip")


# Fixture to create a Dash app for testing
@pytest.fixture
def app(algo: Any) -> Dash:
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
    app = Dash(__name__)

    # Set up a minimal layout with required components
    app.layout = html.Div([
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content"),
    ])

    # Get the full layout from app_layouts
    layout = app_layouts(algo)

    # Register all callbacks
    get_callbacks(app, algo=algo, layout=layout)

    return app
