"""
Tests for the app module.

This module contains tests for the app.py module, which
provides the main entry point for the investment funnel dashboard.
"""



from funnel.app import server


def test_algo_initialization(algo):
    """
    Test that the algo global variable is properly initialized.

    This test verifies that the algo global variable is an instance
    of the investment bot with the expected attributes.
    """
    assert algo is not None, "algo should be initialized"
    assert hasattr(algo, "min_date"), "algo should have min_date attribute"
    assert hasattr(algo, "max_date"), "algo should have max_date attribute"
    assert hasattr(algo, "names"), "algo should have names attribute"


def test_server():
    """
    Test that the server global variable is properly initialized.

    This test verifies that the server global variable is a Flask server
    instance with the expected attributes.
    """
    assert server is not None, "server should be initialized"
    assert hasattr(server, "url_map"), "server should have url_map attribute"
