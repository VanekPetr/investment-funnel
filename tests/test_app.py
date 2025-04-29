import pytest
from dash import Dash, html

from funnel.app import create_app


# Fixture to create the Dash app for testing
@pytest.fixture
def app(algo):
    """Fixture to create the Dash app for testing."""
    return create_app(algo=algo)

# Test the app layout
def test_app_layout(app):
    """Test if the app layout is correctly rendered."""
    assert isinstance(app, Dash)  # Ensure the app is a Dash instance
    assert isinstance(app.layout, html.Div)  # Ensure the layout is a Div component

    # Check for specific components in the layout
    # assert "Your Expected Layout Content" in str(app.layout)
