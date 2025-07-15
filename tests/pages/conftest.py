# tests/conftest.py
from unittest.mock import MagicMock

import dash


def pytest_configure(config):
    """Called before any test module is imported."""
    print("Configuring Dash")
    dash.register_page = MagicMock()
