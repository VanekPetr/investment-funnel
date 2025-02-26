from pathlib import Path

import pytest
from dash import Dash, dcc, html
from ifunnel.models.main import initialize_bot

from funnel.dashboard.app_callbacks import get_callbacks


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "resources"

@pytest.fixture(scope="session", name="algo")
def algo_fixture(resource_dir):
    return initialize_bot(resource_dir / "all_etfs_rets.parquet.gzip")

# Fixture to create a Dash app for testing
@pytest.fixture
def app():
    app = Dash(__name__)
    app.layout = html.Div([
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content"),
    ])
    get_callbacks(app)
    return app
