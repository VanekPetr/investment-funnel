from typing import Any

import pytest
from dash import Dash, html

from funnel.app import create_app


# Fixture to create the Dash app for testing
@pytest.fixture
def app(algo: Any) -> Dash:
    """
    Create a Dash application instance for testing.

    This fixture creates a Dash app using the create_app function
    from the funnel.app module, passing in the algo fixture.

    Args:
        algo: The algorithm bot instance from the algo fixture

    Returns:
        Dash: Configured Dash application instance
    """
    return create_app(algo=algo)


# Test the app layout
def test_app_layout(app: Dash) -> None:
    """
    Test if the app layout is correctly rendered.

    This test verifies that the app is a valid Dash instance and
    that its layout is a Div component as expected. It ensures
    the basic structure of the app is correct.

    Args:
        app: The Dash application instance from the app fixture
    """
    # Ensure the app is a Dash instance
    assert isinstance(app, Dash)

    # Ensure the layout is a Div component
    assert isinstance(app.layout, html.Div)

    # Check for specific components in the layout
    # assert "Your Expected Layout Content" in str(app.layout)
