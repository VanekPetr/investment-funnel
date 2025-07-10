from dash import Dash


def test_callback(app: Dash) -> None:
    """
    Test that all pages of the application are accessible.

    This test verifies that the different pages of the application
    return a 200 status code when accessed, indicating that they
    are properly rendered and don't cause any server errors.

    Args:
        app: The Dash application instance from the app fixture
    """
    # Access the Flask app from the Dash app
    flask_app = app.server

    with flask_app.test_client() as client:
        # Simulate a request to the AI feature selection page
        response = client.get("/page-1")
        assert response.status_code == 200, "AI feature selection page should be accessible"

        # Simulate a request to the backtesting page
        response = client.get("/page-2")
        assert response.status_code == 200, "Backtesting page should be accessible"

        # Simulate a request to the lifecycle investments page
        response = client.get("/page-3")
        assert response.status_code == 200, "Lifecycle investments page should be accessible"
