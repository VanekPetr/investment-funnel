def test_callback(app):
    # Access the Flask app from the Dash app
    flask_app = app.server

    with flask_app.test_client() as client:
        # Simulate a request to /page-1
        response = client.get("/page-1")
        assert response.status_code == 200

        response = client.get("/page-2")
        assert response.status_code == 200

        response = client.get("/page-3")
        assert response.status_code == 200
