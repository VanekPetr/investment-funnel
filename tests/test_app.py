"""
Tests for the app module.

This module contains tests for the app.py module, which
provides the main entry point for the investment funnel dashboard.
"""

from unittest.mock import MagicMock, patch

from dash import dcc, html

from funnel.app import algo, create_app, load_page, main, server


def test_algo_initialization():
    """
    Test that the algo global variable is properly initialized.

    This test verifies that the algo global variable is an instance
    of the investment bot with the expected attributes.
    """
    assert algo is not None, "algo should be initialized"
    assert hasattr(algo, "min_date"), "algo should have min_date attribute"
    assert hasattr(algo, "max_date"), "algo should have max_date attribute"
    assert hasattr(algo, "names"), "algo should have names attribute"


def test_load_page():
    """
    Test that load_page returns the expected layout.

    This test verifies that the load_page function returns a Div
    with the expected structure and components.
    """
    # Call the function
    layout = load_page()

    # Verify the results
    assert layout is not None, "Layout should be created"
    assert isinstance(layout, html.Div), "Layout should be a Div"

    # Check that the layout has the expected structure
    children = layout.children
    assert len(children) > 0, "Layout should have children"

    # Check for essential components
    assert any(isinstance(child, dcc.Location) for child in children), "Layout should have a Location component"
    assert any(isinstance(child, html.Div) and child.id == "page-content" for child in children), \
        "Layout should have a page-content Div"

    # Check for dcc.Store components
    store_ids = [
        "saved-start-date-page-0", "saved-end-date-page-0", "saved-find-fund",
        "saved-top-performers-names", "saved-top-performers", "saved-combine-top-performers",
        "saved-top-performers-pct", "saved-figure-page-0"
    ]
    for store_id in store_ids:
        assert any(isinstance(child, dcc.Store) and child.id == store_id for child in children), \
            f"Layout should have a {store_id} Store"


def test_create_app():
    """
    Test that create_app returns a Flask server instance.

    This test verifies that the create_app function initializes a Dash app
    with the expected configuration and returns its server attribute.
    """
    # Mock the get_callbacks function to avoid side effects
    with patch("funnel.app.get_callbacks") as mock_get_callbacks:
        # Call the function
        flask_server = create_app()

        # Verify the results
        assert flask_server is not None, "Flask server should be created"
        assert hasattr(flask_server, "url_map"), "Flask server should have url_map attribute"

        # Verify that get_callbacks was called
        mock_get_callbacks.assert_called_once()


def test_main():
    """
    Test that main initializes and runs a Dash app.

    This test verifies that the main function creates a Flask server,
    initializes a Dash app with that server, and runs the app.
    """
    # Mock the create_app function to avoid creating a real server
    with patch("funnel.app.create_app") as mock_create_app:
        # Mock the Dash class to avoid creating a real app
        with patch("funnel.app.dash.Dash") as mock_dash:
            # Mock the run method to avoid actually running the app
            mock_app = MagicMock()
            mock_dash.return_value = mock_app

            # Call the function
            main()

            # Verify the results
            mock_create_app.assert_called_once()
            mock_dash.assert_called_once()
            mock_app.run.assert_called_once()


def test_server():
    """
    Test that the server global variable is properly initialized.

    This test verifies that the server global variable is a Flask server
    instance with the expected attributes.
    """
    assert server is not None, "server should be initialized"
    assert hasattr(server, "url_map"), "server should have url_map attribute"
