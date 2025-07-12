"""
Tests for the app_callbacks module.

This module contains tests for the app_callbacks.py module, which
provides the callback functions for the investment funnel dashboard.
"""

import pytest
from unittest.mock import patch, MagicMock

import flask
from dash import html
from dash.dependencies import Input, Output, State

from funnel.dashboard.app_callbacks import get_callbacks


@pytest.fixture
def mock_app():
    """
    Create a mock Dash app for testing callbacks.

    This fixture creates a mock Dash app that can be used to test
    the callback functions in the app_callbacks module.

    Returns:
        MagicMock: A mock Dash app
    """
    app = MagicMock()
    app.callback = MagicMock(return_value=lambda f: f)
    return app


@pytest.fixture
def mock_algo():
    """
    Create a mock algorithm object for testing callbacks.

    This fixture creates a mock algorithm object that can be used
    to test the callback functions in the app_callbacks module.

    Returns:
        MagicMock: A mock algorithm object
    """
    algo = MagicMock()
    algo.min_date = "2013-01-01"
    algo.max_date = "2023-01-01"
    algo.names = ["Asset1", "Asset2", "Asset3"]
    return algo


def test_get_callbacks_registers_callbacks(mock_app, mock_algo):
    """
    Test that get_callbacks registers the expected callbacks.

    This test verifies that the get_callbacks function registers
    all the expected callbacks with the Dash app.

    Args:
        mock_app: A mock Dash app
        mock_algo: A mock algorithm object
    """
    # Call the function
    get_callbacks(mock_app, mock_algo)

    # Verify that callbacks were registered
    assert mock_app.callback.call_count > 0, "No callbacks were registered"

    # Check for specific callbacks
    callback_names = [
        "display_page",
        "plot_lifecycle",
        "plot_backtest",
        "update_output",
        "update_output_lifecycle",
        "show_hide_element",
        "update_output_cluster",
        "update_output_ml_type",
        "update_output_cluster_lifecycle",
        "update_output_top_performers",
        "update_output_ml_type_lifecycle",
        "update_trading_sizes",
        "update_test_date",
        "plot_ml",
        "plot_dots",
    ]

    # Extract the callback functions from the mock calls
    registered_callbacks = []
    for call in mock_app.callback.call_args_list:
        # The callback function is the decorated function, which is passed as the first argument to the decorator
        # In the mock, this is stored in the 'args' attribute of the call object
        if len(call[1]) > 0 and 'callback' in call[1]:
            callback_func = call[1]['callback']
            if hasattr(callback_func, '__name__'):
                registered_callbacks.append(callback_func.__name__)

    # Check that all expected callbacks are registered
    for name in callback_names:
        assert name in registered_callbacks, f"Callback {name} was not registered"


def test_display_page_routing():
    """
    Test that display_page routes to the correct page based on pathname.

    This test verifies that the display_page function returns the
    correct layout based on the pathname.
    """
    # Create mock app and algo
    mock_app = MagicMock()
    mock_algo = MagicMock()

    # Mock the page layout functions
    mock_page_1 = MagicMock(return_value=html.Div(id="mock-page-1"))
    mock_page_2 = MagicMock(return_value=html.Div(id="mock-page-2"))
    mock_page_3 = MagicMock(return_value=html.Div(id="mock-page-3"))
    mock_page_4 = MagicMock(return_value=html.Div(id="mock-page-4"))
    mock_page_mobile = html.Div(id="mock-page-mobile")

    # Mock the flask request
    mock_request = MagicMock()
    mock_request.headers.get.return_value = "desktop"

    # Mock the app_layouts module and flask.request
    with patch("funnel.dashboard.app_callbacks.page_1_layout", mock_page_1), \
         patch("funnel.dashboard.app_callbacks.page_2_layout", mock_page_2), \
         patch("funnel.dashboard.app_callbacks.page_3_layout", mock_page_3), \
         patch("funnel.dashboard.app_callbacks.page_4_layout", mock_page_4), \
         patch("funnel.dashboard.app_callbacks.page_mobile_layout", mock_page_mobile), \
         patch("funnel.dashboard.app_callbacks.flask.request", mock_request):

        # Get the display_page function
        callbacks = get_callbacks(mock_app, mock_algo)
        display_page = None
        for call in mock_app.callback.call_args_list:
            if len(call[1]) > 0 and 'callback' in call[1]:
                callback_func = call[1]['callback']
                if hasattr(callback_func, '__name__') and callback_func.__name__ == "display_page":
                    display_page = callback_func
                    break

        assert display_page is not None, "display_page function not found"

        # Test routing to page 1 (root)
        result = display_page("/")
        assert result.id == "mock-page-1", "Should route to page 1 for /"
        mock_page_1.assert_called_once_with(mock_algo)

        # Reset mock calls
        mock_page_1.reset_mock()

        # Test routing to page 2
        result = display_page("/page-1")
        assert result.id == "mock-page-2", "Should route to page 2 for /page-1"
        mock_page_2.assert_called_once_with(mock_algo)

        # Test routing to page 3
        result = display_page("/page-2")
        assert result.id == "mock-page-3", "Should route to page 3 for /page-2"
        mock_page_3.assert_called_once_with(mock_algo)

        # Test routing to page 4 (default)
        result = display_page("/unknown")
        assert result.id == "mock-page-4", "Should route to page 4 for unknown pathname"
        mock_page_4.assert_called_once_with(mock_algo)

        # Test routing to mobile page
        mock_request.headers.get.return_value = "mobile"
        result = display_page("/")
        assert result.id == "mock-page-mobile", "Should route to mobile page for mobile user agent"


def test_update_output():
    """
    Test that update_output returns the expected output.

    This test verifies that the update_output function returns
    the expected string based on the input value.
    """
    # Create mock app and algo
    mock_app = MagicMock()
    mock_algo = MagicMock()

    # Get the update_output function
    callbacks = get_callbacks(mock_app, mock_algo)
    update_output = None
    for call in mock_app.callback.call_args_list:
        if len(call[1]) > 0 and 'callback' in call[1]:
            callback_func = call[1]['callback']
            if hasattr(callback_func, '__name__') and callback_func.__name__ == "update_output":
                update_output = callback_func
                break

    assert update_output is not None, "update_output function not found"

    # Test with different values
    assert update_output(500) == "# of scenarios: 500", "Should return correct string for 500"
    assert update_output(1000) == "# of scenarios: 1000", "Should return correct string for 1000"


def test_update_output_lifecycle():
    """
    Test that update_output_lifecycle returns the expected output.

    This test verifies that the update_output_lifecycle function returns
    the expected string based on the input value.
    """
    # Create mock app and algo
    mock_app = MagicMock()
    mock_algo = MagicMock()

    # Get the update_output_lifecycle function
    callbacks = get_callbacks(mock_app, mock_algo)
    update_output_lifecycle = None
    for call in mock_app.callback.call_args_list:
        if len(call[1]) > 0 and 'callback' in call[1]:
            callback_func = call[1]['callback']
            if hasattr(callback_func, '__name__') and callback_func.__name__ == "update_output_lifecycle":
                update_output_lifecycle = callback_func
                break

    assert update_output_lifecycle is not None, "update_output_lifecycle function not found"

    # Test with different values
    assert update_output_lifecycle(500) == "# of scenarios: 500", "Should return correct string for 500"
    assert update_output_lifecycle(1000) == "# of scenarios: 1000", "Should return correct string for 1000"


def test_show_hide_element():
    """
    Test that show_hide_element returns the expected output.

    This test verifies that the show_hide_element function returns
    the expected style dictionary based on the input value.
    """
    # Create mock app and algo
    mock_app = MagicMock()
    mock_algo = MagicMock()

    # Get the show_hide_element function
    callbacks = get_callbacks(mock_app, mock_algo)
    show_hide_element = None
    for call in mock_app.callback.call_args_list:
        if len(call[1]) > 0 and 'callback' in call[1]:
            callback_func = call[1]['callback']
            if hasattr(callback_func, '__name__') and callback_func.__name__ == "show_hide_element":
                show_hide_element = callback_func
                break

    assert show_hide_element is not None, "show_hide_element function not found"

    # Test with different values
    assert show_hide_element("ECOS_BB") == {"display": "block"}, "Should show for ECOS_BB"
    assert show_hide_element("CPLEX") == {"display": "block"}, "Should show for CPLEX"
    assert show_hide_element("MOSEK") == {"display": "block"}, "Should show for MOSEK"
    assert show_hide_element("ECOS") == {"display": "none"}, "Should hide for ECOS"
    assert show_hide_element("SCS") == {"display": "none"}, "Should hide for SCS"


def test_update_output_cluster():
    """
    Test that update_output_cluster returns the expected output.

    This test verifies that the update_output_cluster function returns
    the expected style dictionary based on the input value.
    """
    # Create mock app and algo
    mock_app = MagicMock()
    mock_algo = MagicMock()

    # Get the update_output_cluster function
    callbacks = get_callbacks(mock_app, mock_algo)
    update_output_cluster = None
    for call in mock_app.callback.call_args_list:
        if len(call[1]) > 0 and 'callback' in call[1]:
            callback_func = call[1]['callback']
            if hasattr(callback_func, '__name__') and callback_func.__name__ == "update_output_cluster":
                update_output_cluster = callback_func
                break

    assert update_output_cluster is not None, "update_output_cluster function not found"

    # Test with different values
    assert update_output_cluster("Clustering") == {"display": "block"}, "Should show for Clustering"
    assert update_output_cluster("MST") == {"display": "none"}, "Should hide for MST"
    assert update_output_cluster(None) == {"display": "none"}, "Should hide for None"


def test_update_output_ml_type():
    """
    Test that update_output_ml_type returns the expected output.

    This test verifies that the update_output_ml_type function returns
    the expected string based on the input value.
    """
    # Create mock app and algo
    mock_app = MagicMock()
    mock_algo = MagicMock()

    # Get the update_output_ml_type function
    callbacks = get_callbacks(mock_app, mock_algo)
    update_output_ml_type = None
    for call in mock_app.callback.call_args_list:
        if len(call[1]) > 0 and 'callback' in call[1]:
            callback_func = call[1]['callback']
            if hasattr(callback_func, '__name__') and callback_func.__name__ == "update_output_ml_type":
                update_output_ml_type = callback_func
                break

    assert update_output_ml_type is not None, "update_output_ml_type function not found"

    # Test with different values
    assert update_output_ml_type(3) == "# of clusters or # of MST runs: 3", "Should return correct string for 3"
    assert update_output_ml_type(5) == "# of clusters or # of MST runs: 5", "Should return correct string for 5"


def test_update_output_cluster_lifecycle():
    """
    Test that update_output_cluster_lifecycle returns the expected output.

    This test verifies that the update_output_cluster_lifecycle function returns
    the expected style dictionary based on the input value.
    """
    # Create mock app and algo
    mock_app = MagicMock()
    mock_algo = MagicMock()

    # Get the update_output_cluster_lifecycle function
    callbacks = get_callbacks(mock_app, mock_algo)
    update_output_cluster_lifecycle = None
    for call in mock_app.callback.call_args_list:
        if len(call[1]) > 0 and 'callback' in call[1]:
            callback_func = call[1]['callback']
            if hasattr(callback_func, '__name__') and callback_func.__name__ == "update_output_cluster_lifecycle":
                update_output_cluster_lifecycle = callback_func
                break

    assert update_output_cluster_lifecycle is not None, "update_output_cluster_lifecycle function not found"

    # Test with different values
    assert update_output_cluster_lifecycle("Clustering") == {"display": "block"}, "Should show for Clustering"
    assert update_output_cluster_lifecycle("MST") == {"display": "none"}, "Should hide for MST"
    assert update_output_cluster_lifecycle(None) == {"display": "none"}, "Should hide for None"


def test_update_output_top_performers():
    """
    Test that update_output_top_performers returns the expected output.

    This test verifies that the update_output_top_performers function returns
    the expected style dictionary based on the input value.
    """
    # Create mock app and algo
    mock_app = MagicMock()
    mock_algo = MagicMock()

    # Get the update_output_top_performers function
    callbacks = get_callbacks(mock_app, mock_algo)
    update_output_top_performers = None
    for call in mock_app.callback.call_args_list:
        if len(call[1]) > 0 and 'callback' in call[1]:
            callback_func = call[1]['callback']
            if hasattr(callback_func, '__name__') and callback_func.__name__ == "update_output_top_performers":
                update_output_top_performers = callback_func
                break

    assert update_output_top_performers is not None, "update_output_top_performers function not found"

    # Test with different values
    assert update_output_top_performers("yes") == {"display": "block"}, "Should show for yes"
    assert update_output_top_performers("no") == {"display": "none"}, "Should hide for no"
    assert update_output_top_performers(None) == {"display": "none"}, "Should hide for None"


def test_update_output_ml_type_lifecycle():
    """
    Test that update_output_ml_type_lifecycle returns the expected output.

    This test verifies that the update_output_ml_type_lifecycle function returns
    the expected string based on the input value.
    """
    # Create mock app and algo
    mock_app = MagicMock()
    mock_algo = MagicMock()

    # Get the update_output_ml_type_lifecycle function
    callbacks = get_callbacks(mock_app, mock_algo)
    update_output_ml_type_lifecycle = None
    for call in mock_app.callback.call_args_list:
        if len(call[1]) > 0 and 'callback' in call[1]:
            callback_func = call[1]['callback']
            if hasattr(callback_func, '__name__') and callback_func.__name__ == "update_output_ml_type_lifecycle":
                update_output_ml_type_lifecycle = callback_func
                break

    assert update_output_ml_type_lifecycle is not None, "update_output_ml_type_lifecycle function not found"

    # Test with different values
    assert update_output_ml_type_lifecycle(3) == "# of clusters or # of MST runs: 3", "Should return correct string for 3"
    assert update_output_ml_type_lifecycle(5) == "# of clusters or # of MST runs: 5", "Should return correct string for 5"


def test_update_trading_sizes():
    """
    Test that update_trading_sizes returns the expected output.

    This test verifies that the update_trading_sizes function returns
    the expected string based on the input value.
    """
    # Create mock app and algo
    mock_app = MagicMock()
    mock_algo = MagicMock()

    # Get the update_trading_sizes function
    callbacks = get_callbacks(mock_app, mock_algo)
    update_trading_sizes = None
    for call in mock_app.callback.call_args_list:
        if len(call[1]) > 0 and 'callback' in call[1]:
            callback_func = call[1]['callback']
            if hasattr(callback_func, '__name__') and callback_func.__name__ == "update_trading_sizes":
                update_trading_sizes = callback_func
                break

    assert update_trading_sizes is not None, "update_trading_sizes function not found"

    # Test with different values
    assert update_trading_sizes(5) == "Minimum required asset weight in the portfolio: 5%", "Should return correct string for 5"
    assert update_trading_sizes(10) == "Minimum required asset weight in the portfolio: 10%", "Should return correct string for 10"


def test_update_test_date():
    """
    Test that update_test_date returns the expected output.

    This test verifies that the update_test_date function returns
    the expected dates based on the input value.
    """
    # Create mock app and algo
    mock_app = MagicMock()
    mock_algo = MagicMock()
    mock_algo.min_date = "2013-01-01"
    mock_algo.max_date = "2023-01-01"

    # Get the update_test_date function
    callbacks = get_callbacks(mock_app, mock_algo)
    update_test_date = None
    for call in mock_app.callback.call_args_list:
        if len(call[1]) > 0 and 'callback' in call[1]:
            callback_func = call[1]['callback']
            if hasattr(callback_func, '__name__') and callback_func.__name__ == "update_test_date":
                update_test_date = callback_func
                break

    assert update_test_date is not None, "update_test_date function not found"

    # Test with a selected date
    result = update_test_date("2018-01-01", "2017-07-01")
    assert len(result) == 5, "Should return 5 values"
    assert result[0] == "2018-01-01", "First value should be the selected date"
    assert result[1] == "2023-01-01", "Second value should be algo.max_date"
    assert result[2] == "2013-01-01", "Third value should be algo.min_date"
    assert result[3] == "2018-01-01", "Fourth value should be the selected date"
    assert result[4] == "2018-01-01", "Fifth value should be the selected date"

    # Test with None as selected date (should use saved_split_date)
    result = update_test_date(None, "2017-07-01")
    assert len(result) == 5, "Should return 5 values"
    assert result[0] == "2017-07-01", "First value should be the saved_split_date"
    assert result[1] == "2023-01-01", "Second value should be algo.max_date"
    assert result[2] == "2013-01-01", "Third value should be algo.min_date"
    assert result[3] == "2017-07-01", "Fourth value should be the saved_split_date"
    assert result[4] == "2017-07-01", "Fifth value should be the saved_split_date"


def test_plot_lifecycle_no_click():
    """
    Test that plot_lifecycle returns the saved values when not clicked.

    This test verifies that the plot_lifecycle function returns
    the saved values when the button is not clicked.
    """
    # Create mock app and algo
    mock_app = MagicMock()
    mock_algo = MagicMock()

    # Get the plot_lifecycle function
    callbacks = get_callbacks(mock_app, mock_algo)
    plot_lifecycle = None
    for call in mock_app.callback.call_args_list:
        if len(call[1]) > 0 and 'callback' in call[1]:
            callback_func = call[1]['callback']
            if hasattr(callback_func, '__name__') and callback_func.__name__ == "plot_lifecycle":
                plot_lifecycle = callback_func
                break

    assert plot_lifecycle is not None, "plot_lifecycle function not found"

    # Create mock saved values
    saved_model = "MST"
    saved_model_spec = 2
    saved_pick_top = 5
    saved_scen_model = "Bootstrap"
    saved_scen_spec = 1000
    saved_portfolio_value = 100000
    saved_yearly_withdraws = 1000
    saved_risk_preference = 15
    saved_end_year = 2040
    saved_glidepaths_figure = html.Div(id="saved-glidepaths")
    saved_performance_figure = html.Div(id="saved-performance")
    saved_lifecycle_all_figure = html.Div(id="saved-lifecycle-all")

    # Call the function with no click
    result = plot_lifecycle(
        None,  # No click
        "MST",  # model
        2,  # model_spec
        5,  # pick_top
        "Bootstrap",  # scen_model
        1000,  # scen_spec
        "2013-01-01",  # start_data
        "2023-01-01",  # end_train
        2040,  # end_year
        100000,  # portfolio_value
        1000,  # yearly_withdraws
        15,  # risk_preference
        saved_model,
        saved_model_spec,
        saved_pick_top,
        saved_scen_model,
        saved_scen_spec,
        saved_portfolio_value,
        saved_yearly_withdraws,
        saved_risk_preference,
        saved_end_year,
        saved_glidepaths_figure,
        saved_performance_figure,
        saved_lifecycle_all_figure,
    )

    # Verify the results
    assert len(result) == 21, "Should return 21 values"
    assert result[0] == saved_glidepaths_figure, "Should return saved glidepaths figure"
    assert result[1] == saved_performance_figure, "Should return saved performance figure"
    assert result[2] == saved_lifecycle_all_figure, "Should return saved lifecycle all figure"
    assert result[3] == saved_model, "Should return saved model"
    assert result[4] == saved_model_spec, "Should return saved model spec"
    assert result[5] == saved_pick_top, "Should return saved pick top"
    assert result[6] == saved_scen_model, "Should return saved scenario model"
    assert result[7] == saved_scen_spec, "Should return saved scenario spec"
    assert result[8] == saved_model, "Should return saved model"
    assert result[9] == saved_model_spec, "Should return saved model spec"
    assert result[10] == saved_pick_top, "Should return saved pick top"
    assert result[11] == saved_scen_model, "Should return saved scenario model"
    assert result[12] == saved_scen_spec, "Should return saved scenario spec"
    assert result[13] == saved_portfolio_value, "Should return saved portfolio value"
    assert result[14] == saved_yearly_withdraws, "Should return saved yearly withdraws"
    assert result[15] == saved_risk_preference, "Should return saved risk preference"
    assert result[16] == saved_end_year, "Should return saved end year"
    assert result[17] == saved_glidepaths_figure, "Should return saved glidepaths figure"
    assert result[18] == saved_performance_figure, "Should return saved performance figure"
    assert result[19] == saved_lifecycle_all_figure, "Should return saved lifecycle all figure"
    assert result[20] is True, "Should return True for loading output"


def test_plot_lifecycle_with_click():
    """
    Test that plot_lifecycle generates new figures when clicked.

    This test verifies that the plot_lifecycle function generates
    new figures when the button is clicked.
    """
    # Create mock app and algo
    mock_app = MagicMock()
    mock_algo = MagicMock()

    # Mock the lifecycle_scenario_analysis method
    mock_fig_glidepaths = {"data": [], "layout": {}}
    mock_fig_performance = {"data": [], "layout": {}}
    mock_fig_composition_all = {"data": [], "layout": {}}
    mock_subset = ["Asset1", "Asset2"]
    mock_algo.lifecycle_scenario_analysis.return_value = (
        mock_fig_glidepaths,
        mock_fig_performance,
        mock_fig_composition_all,
        mock_subset
    )

    # Mock the mst method
    mock_algo.mst.return_value = (None, ["Asset1", "Asset2"])

    # Get the plot_lifecycle function
    callbacks = get_callbacks(mock_app, mock_algo)
    plot_lifecycle = None
    for call in mock_app.callback.call_args_list:
        if len(call[1]) > 0 and 'callback' in call[1]:
            callback_func = call[1]['callback']
            if hasattr(callback_func, '__name__') and callback_func.__name__ == "plot_lifecycle":
                plot_lifecycle = callback_func
                break

    assert plot_lifecycle is not None, "plot_lifecycle function not found"

    # Create mock saved values
    saved_model = "MST"
    saved_model_spec = 2
    saved_pick_top = 5
    saved_scen_model = "Bootstrap"
    saved_scen_spec = 1000
    saved_portfolio_value = 100000
    saved_yearly_withdraws = 1000
    saved_risk_preference = 15
    saved_end_year = 2040
    saved_glidepaths_figure = html.Div(id="saved-glidepaths")
    saved_performance_figure = html.Div(id="saved-performance")
    saved_lifecycle_all_figure = html.Div(id="saved-lifecycle-all")

    # Call the function with a click
    result = plot_lifecycle(
        1,  # Click
        "MST",  # model
        2,  # model_spec
        5,  # pick_top
        "Bootstrap",  # scen_model
        1000,  # scen_spec
        "2013-01-01",  # start_data
        "2023-01-01",  # end_train
        2040,  # end_year
        100000,  # portfolio_value
        1000,  # yearly_withdraws
        15,  # risk_preference
        saved_model,
        saved_model_spec,
        saved_pick_top,
        saved_scen_model,
        saved_scen_spec,
        saved_portfolio_value,
        saved_yearly_withdraws,
        saved_risk_preference,
        saved_end_year,
        saved_glidepaths_figure,
        saved_performance_figure,
        saved_lifecycle_all_figure,
    )

    # Verify the results
    assert len(result) == 21, "Should return 21 values"
    assert result[0] is not None, "Should return a glidepaths figure"
    assert result[1] is not None, "Should return a performance figure"
    assert result[2] is not None, "Should return a lifecycle all figure"
    assert result[3] == "MST", "Should return the input model"
    assert result[4] == 2, "Should return the input model spec"
    assert result[5] == 5, "Should return the input pick top"
    assert result[6] == "Bootstrap", "Should return the input scenario model"
    assert result[7] == 1000, "Should return the input scenario spec"
    assert result[8] == "MST", "Should return the input model"
    assert result[9] == 2, "Should return the input model spec"
    assert result[10] == 5, "Should return the input pick top"
    assert result[11] == "Bootstrap", "Should return the input scenario model"
    assert result[12] == 1000, "Should return the input scenario spec"
    assert result[13] == 100000, "Should return the input portfolio value"
    assert result[14] == 1000, "Should return the input yearly withdraws"
    assert result[15] == 15, "Should return the input risk preference"
    assert result[16] == 2040, "Should return the input end year"
    assert result[17] is not None, "Should return a glidepaths figure"
    assert result[18] is not None, "Should return a performance figure"
    assert result[19] is not None, "Should return a lifecycle all figure"
    assert result[20] is True, "Should return True for loading output"

    # Verify that the algo methods were called with the correct parameters
    mock_algo.mst.assert_called_once_with(
        start_date="2013-01-01",
        end_date="2023-01-01",
        n_mst_runs=2,
        plot=False,
    )
    mock_algo.lifecycle_scenario_analysis.assert_called_once_with(
        start_date="2013-01-01",
        end_date="2023-01-01",
        subset=["Asset1", "Asset2"],
        scenario_model="Bootstrap",
        n_scenarios=1000,
        portfolio_value=100000,
        yearly_withdraws=1000,
        risk_preference=15,
        end_year=2040,
    )


def test_plot_backtest_no_click():
    """
    Test that plot_backtest returns the saved values when not clicked.

    This test verifies that the plot_backtest function returns
    the saved values when the button is not clicked.
    """
    # Create mock app and algo
    mock_app = MagicMock()
    mock_algo = MagicMock()

    # Get the plot_backtest function
    callbacks = get_callbacks(mock_app, mock_algo)
    plot_backtest = None
    for call in mock_app.callback.call_args_list:
        if len(call[1]) > 0 and 'callback' in call[1]:
            callback_func = call[1]['callback']
            if hasattr(callback_func, '__name__') and callback_func.__name__ == "plot_backtest":
                plot_backtest = callback_func
                break

    assert plot_backtest is not None, "plot_backtest function not found"

    # Create mock saved values
    saved_model = "MST"
    saved_model_spec = 2
    saved_pick_top = 5
    saved_scen_model = "Bootstrap"
    saved_scen_spec = 1000
    saved_benchmark = ["Benchmark1"]
    saved_perf_figure = html.Div(id="saved-perf")
    saved_comp_figure = html.Div(id="saved-comp")
    saved_universe_figure = html.Div(id="saved-universe")
    saved_solver = "ECOS"
    saved_optimization_model = "CVaR model"

    # Call the function with no click
    result = plot_backtest(
        None,  # No click
        "MST",  # model
        2,  # model_spec
        5,  # pick_top
        "Bootstrap",  # scen_model
        1000,  # scen_spec
        ["Benchmark1"],  # benchmark
        "2013-01-01",  # start_data
        "2017-01-01",  # end_train
        "2017-01-01",  # start_test
        "2023-01-01",  # end_data
        saved_model,
        saved_model_spec,
        saved_pick_top,
        saved_scen_model,
        saved_scen_spec,
        saved_benchmark,
        saved_perf_figure,
        saved_comp_figure,
        saved_universe_figure,
        "ECOS",  # solver
        saved_solver,
        "CVaR model",  # optimization_model
        saved_optimization_model,
        0,  # lower_bound
    )

    # Verify the results
    assert len(result) == 12, "Should return 12 values"
    assert result[0] == saved_perf_figure, "Should return saved perf figure"
    assert result[1] == saved_comp_figure, "Should return saved comp figure"
    assert result[2] == saved_model, "Should return saved model"
    assert result[3] == saved_model_spec, "Should return saved model spec"
    assert result[4] == saved_pick_top, "Should return saved pick top"
    assert result[5] == saved_scen_model, "Should return saved scenario model"
    assert result[6] == saved_scen_spec, "Should return saved scenario spec"
    assert result[7] == saved_benchmark, "Should return saved benchmark"
    assert result[8] is True, "Should return True for loading output"
    assert result[9] == saved_universe_figure, "Should return saved universe figure"
    assert result[10] == saved_solver, "Should return saved solver"
    assert result[11] == saved_optimization_model, "Should return saved optimization model"


def test_plot_backtest_with_click():
    """
    Test that plot_backtest generates new figures when clicked.

    This test verifies that the plot_backtest function generates
    new figures when the button is clicked.
    """
    # Create mock app and algo
    mock_app = MagicMock()
    mock_algo = MagicMock()

    # Mock the backtest method
    mock_opt_table = MagicMock()
    mock_opt_table.iloc = [MagicMock(to_list=lambda: [1, 2, 3])]
    mock_bench_table = MagicMock()
    mock_bench_table.iloc = [MagicMock(to_list=lambda: [4, 5, 6])]
    mock_fig_performance = {"data": [], "layout": {}}
    mock_fig_composition = {"data": [], "layout": {}}
    mock_algo.backtest.return_value = (
        mock_opt_table,
        mock_bench_table,
        mock_fig_performance,
        mock_fig_composition
    )

    # Mock the plot_dots method
    mock_fig_dots = {"data": [], "layout": {}}
    mock_algo.plot_dots.return_value = mock_fig_dots

    # Mock the mst method
    mock_algo.mst.return_value = (None, ["Asset1", "Asset2"])

    # Get the plot_backtest function
    callbacks = get_callbacks(mock_app, mock_algo)
    plot_backtest = None
    for call in mock_app.callback.call_args_list:
        if len(call[1]) > 0 and 'callback' in call[1]:
            callback_func = call[1]['callback']
            if hasattr(callback_func, '__name__') and callback_func.__name__ == "plot_backtest":
                plot_backtest = callback_func
                break

    assert plot_backtest is not None, "plot_backtest function not found"

    # Create mock saved values
    saved_model = "MST"
    saved_model_spec = 2
    saved_pick_top = 5
    saved_scen_model = "Bootstrap"
    saved_scen_spec = 1000
    saved_benchmark = ["Benchmark1"]
    saved_perf_figure = html.Div(id="saved-perf")
    saved_comp_figure = html.Div(id="saved-comp")
    saved_universe_figure = html.Div(id="saved-universe")
    saved_solver = "ECOS"
    saved_optimization_model = "CVaR model"

    # Call the function with a click
    result = plot_backtest(
        1,  # Click
        "MST",  # model
        2,  # model_spec
        5,  # pick_top
        "Bootstrap",  # scen_model
        1000,  # scen_spec
        ["Benchmark1"],  # benchmark
        "2013-01-01",  # start_data
        "2017-01-01",  # end_train
        "2017-01-01",  # start_test
        "2023-01-01",  # end_data
        saved_model,
        saved_model_spec,
        saved_pick_top,
        saved_scen_model,
        saved_scen_spec,
        saved_benchmark,
        saved_perf_figure,
        saved_comp_figure,
        saved_universe_figure,
        "ECOS",  # solver
        saved_solver,
        "CVaR model",  # optimization_model
        saved_optimization_model,
        0,  # lower_bound
    )

    # Verify the results
    assert len(result) == 12, "Should return 12 values"
    assert result[0] is not None, "Should return a perf figure"
    assert result[1] is not None, "Should return a comp figure"
    assert result[2] == "MST", "Should return the input model"
    assert result[3] == 2, "Should return the input model spec"
    assert result[4] == 5, "Should return the input pick top"
    assert result[5] == "Bootstrap", "Should return the input scenario model"
    assert result[6] == 1000, "Should return the input scenario spec"
    assert result[7] == ["Benchmark1"], "Should return the input benchmark"
    assert result[8] is True, "Should return True for loading output"
    assert result[9] is not None, "Should return a universe figure"
    assert result[10] == "ECOS", "Should return the input solver"
    assert result[11] == "CVaR model", "Should return the input optimization model"

    # Verify that the algo methods were called with the correct parameters
    mock_algo.mst.assert_called_once_with(
        start_date="2013-01-01",
        end_date="2017-01-01",
        n_mst_runs=2,
        plot=False,
    )
    mock_algo.backtest.assert_called_once_with(
        start_train_date="2013-01-01",
        end_train_date="2017-01-01",
        start_test_date="2017-01-01",
        end_test_date="2023-01-01",
        subset=["Asset1", "Asset2"],
        benchmarks=["Benchmark1"],
        scenario_model="Bootstrap",
        n_scenarios=1000,
        optimization_model="CVaR model",
        solver="ECOS",
        lower_bound=0,
    )
    mock_algo.plot_dots.assert_called_once()


def test_plot_ml_no_click():
    """
    Test that plot_ml returns the saved values when not clicked.

    This test verifies that the plot_ml function returns
    the saved values when the button is not clicked.
    """
    # Create mock app and algo
    mock_app = MagicMock()
    mock_algo = MagicMock()

    # Get the plot_ml function
    callbacks = get_callbacks(mock_app, mock_algo)
    plot_ml = None
    for call in mock_app.callback.call_args_list:
        if len(call[1]) > 0 and 'callback' in call[1]:
            callback_func = call[1]['callback']
            if hasattr(callback_func, '__name__') and callback_func.__name__ == "plot_ml":
                plot_ml = callback_func
                break

    assert plot_ml is not None, "plot_ml function not found"

    # Create mock saved values
    saved_start = "2013-01-01"
    saved_end = "2023-01-01"
    saved_ai_table = [{"Name": "Asset1", "ISIN": "123456"}]
    saved_model = "MST"
    saved_spec = 2
    saved_text = "Number of selected assets: 1"
    saved_figure = html.Div(id="saved-ml-figure")

    # Call the function with no click
    result = plot_ml(
        "/page-1",  # pathname
        None,  # No click
        "MST",  # model
        2,  # spec
        "2013-01-01",  # start
        "2023-01-01",  # end
        saved_start,
        saved_end,
        saved_ai_table,
        saved_model,
        saved_spec,
        saved_text,
        saved_figure,
    )

    # Verify the results
    assert len(result) == 15, "Should return 15 values"
    assert result[0] == saved_figure, "Should return saved figure"
    assert result[1] == saved_start, "Should return saved start date"
    assert result[2] == saved_end, "Should return saved end date"
    assert result[3] == saved_ai_table, "Should return saved AI table"
    assert result[4] == saved_text, "Should return saved text"
    assert result[5] == saved_model, "Should return saved model"
    assert result[6] == saved_spec, "Should return saved spec"
    assert result[7] == saved_start, "Should return saved start date"
    assert result[8] == saved_end, "Should return saved end date"
    assert result[9] == saved_ai_table, "Should return saved AI table"
    assert result[10] == saved_model, "Should return saved model"
    assert result[11] == saved_spec, "Should return saved spec"
    assert result[12] == saved_text, "Should return saved text"
    assert result[13] == saved_figure, "Should return saved figure"
    assert result[14] is True, "Should return True for loading output"


def test_plot_ml_with_click():
    """
    Test that plot_ml generates new figures when clicked.

    This test verifies that the plot_ml function generates
    new figures when the button is clicked.
    """
    # Create mock app and algo
    mock_app = MagicMock()
    mock_algo = MagicMock()

    # Mock the mst method
    mock_fig = {"data": [], "layout": {}}
    mock_subset = ["Asset1", "Asset2"]
    mock_algo.mst.return_value = (mock_fig, mock_subset)

    # Mock the get_stat method
    mock_stat = MagicMock()
    mock_stat.loc = MagicMock()
    mock_stat.loc.return_value = MagicMock()
    mock_stat.loc.return_value.__getitem__ = MagicMock(return_value=MagicMock(round=lambda x: [1.0, 2.0]))
    mock_algo.get_stat.return_value = mock_stat

    # Get the plot_ml function
    callbacks = get_callbacks(mock_app, mock_algo)
    plot_ml = None
    for call in mock_app.callback.call_args_list:
        if len(call[1]) > 0 and 'callback' in call[1]:
            callback_func = call[1]['callback']
            if hasattr(callback_func, '__name__') and callback_func.__name__ == "plot_ml":
                plot_ml = callback_func
                break

    assert plot_ml is not None, "plot_ml function not found"

    # Create mock saved values
    saved_start = "2013-01-01"
    saved_end = "2023-01-01"
    saved_ai_table = [{"Name": "Asset1", "ISIN": "123456"}]
    saved_model = "MST"
    saved_spec = 2
    saved_text = "Number of selected assets: 1"
    saved_figure = html.Div(id="saved-ml-figure")

    # Call the function with a click
    result = plot_ml(
        "/page-1",  # pathname
        1,  # Click
        "MST",  # model
        2,  # spec
        "2013-01-01",  # start
        "2023-01-01",  # end
        saved_start,
        saved_end,
        saved_ai_table,
        saved_model,
        saved_spec,
        saved_text,
        saved_figure,
    )

    # Verify the results
    assert len(result) == 15, "Should return 15 values"
    assert result[0] is not None, "Should return a figure"
    assert result[1] == "2013-01-01", "Should return the input start date"
    assert result[2] == "2023-01-01", "Should return the input end date"
    assert result[3] is not None, "Should return an AI table"
    assert "Number of selected assets:" in result[4], "Should return a text with the number of selected assets"
    assert result[5] == "MST", "Should return the input model"
    assert result[6] == 2, "Should return the input spec"
    assert result[7] == "2013-01-01", "Should return the input start date"
    assert result[8] == "2023-01-01", "Should return the input end date"
    assert result[9] is not None, "Should return an AI table"
    assert result[10] == "MST", "Should return the input model"
    assert result[11] == 2, "Should return the input spec"
    assert "Number of selected assets:" in result[12], "Should return a text with the number of selected assets"
    assert result[13] is not None, "Should return a figure"
    assert result[14] is True, "Should return True for loading output"

    # Verify that the algo methods were called with the correct parameters
    mock_algo.mst.assert_called_once_with(
        start_date="2013-01-01",
        end_date="2023-01-01",
        n_mst_runs=2,
        plot=True,
    )
    mock_algo.get_stat.assert_called_once_with(
        start_date="2013-01-01",
        end_date="2023-01-01",
    )


def test_plot_dots_no_click():
    """
    Test that plot_dots returns the saved values when not clicked.

    This test verifies that the plot_dots function returns
    the saved values when the button is not clicked.
    """
    # Create mock app and algo
    mock_app = MagicMock()
    mock_algo = MagicMock()

    # Get the plot_dots function
    callbacks = get_callbacks(mock_app, mock_algo)
    plot_dots = None
    for call in mock_app.callback.call_args_list:
        if len(call[1]) > 0 and 'callback' in call[1]:
            callback_func = call[1]['callback']
            if hasattr(callback_func, '__name__') and callback_func.__name__ == "plot_dots":
                plot_dots = callback_func
                break

    assert plot_dots is not None, "plot_dots function not found"

    # Create mock saved values
    saved_figure = html.Div(id="saved-dots-figure")
    saved_start = "2013-01-01"
    saved_end = "2023-01-01"
    saved_find_fund = ["Fund1", "Fund2"]
    saved_top_performers_names = ["Fund3", "Fund4"]
    saved_top_performers = "yes"
    saved_combine_top_performers = "no"
    saved_top_performers_pct = 15

    # Call the function with no click
    result = plot_dots(
        "trigger",  # trigger
        None,  # No click
        "2013-01-01",  # start
        "2023-01-01",  # end
        ["Fund1", "Fund2"],  # search
        saved_start,
        saved_end,
        saved_find_fund,
        saved_figure,
        "yes",  # top_performers
        "no",  # combine_top_performers
        15,  # top_performers_pct
        saved_top_performers_names,
        saved_top_performers,
        saved_combine_top_performers,
        saved_top_performers_pct,
    )

    # Verify the results
    assert len(result) == 16, "Should return 16 values"
    assert result[0] == saved_figure, "Should return saved figure"
    assert result[1] == saved_start, "Should return saved start date"
    assert result[2] == saved_end, "Should return saved end date"
    assert result[3] == saved_find_fund, "Should return saved find fund"
    assert result[4] == saved_start, "Should return saved start date"
    assert result[5] == saved_end, "Should return saved end date"
    assert result[6] == saved_find_fund, "Should return saved find fund"
    assert result[7] == saved_figure, "Should return saved figure"
    assert result[8] is True, "Should return True for loading output"
    assert result[9] == saved_top_performers_names, "Should return saved top performers names"
    assert result[10] == saved_top_performers, "Should return saved top performers"
    assert result[11] == saved_combine_top_performers, "Should return saved combine top performers"
    assert result[12] == saved_top_performers_pct, "Should return saved top performers percentage"
    assert result[13] == saved_top_performers, "Should return saved top performers"
    assert result[14] == saved_combine_top_performers, "Should return saved combine top performers"
    assert result[15] == saved_top_performers_pct, "Should return saved top performers percentage"


def test_plot_dots_with_click():
    """
    Test that plot_dots generates new figures when clicked.

    This test verifies that the plot_dots function generates
    new figures when the button is clicked.
    """
    # Create mock app and algo
    mock_app = MagicMock()
    mock_algo = MagicMock()

    # Mock the get_top_performing_assets method
    mock_algo.get_top_performing_assets.return_value = ["Fund3", "Fund4"]

    # Mock the plot_dots method
    mock_fig = {"data": [], "layout": {}}
    mock_algo.plot_dots.return_value = mock_fig

    # Get the plot_dots function
    callbacks = get_callbacks(mock_app, mock_algo)
    plot_dots = None
    for call in mock_app.callback.call_args_list:
        if len(call[1]) > 0 and 'callback' in call[1]:
            callback_func = call[1]['callback']
            if hasattr(callback_func, '__name__') and callback_func.__name__ == "plot_dots":
                plot_dots = callback_func
                break

    assert plot_dots is not None, "plot_dots function not found"

    # Create mock saved values
    saved_figure = html.Div(id="saved-dots-figure")
    saved_start = "2013-01-01"
    saved_end = "2023-01-01"
    saved_find_fund = ["Fund1", "Fund2"]
    saved_top_performers_names = ["Fund5", "Fund6"]
    saved_top_performers = "no"
    saved_combine_top_performers = "no"
    saved_top_performers_pct = 10

    # Call the function with a click and top_performers="yes"
    result = plot_dots(
        "trigger",  # trigger
        1,  # Click
        "2013-01-01",  # start
        "2023-01-01",  # end
        ["Fund1", "Fund2"],  # search
        saved_start,
        saved_end,
        saved_find_fund,
        saved_figure,
        "yes",  # top_performers
        "no",  # combine_top_performers
        15,  # top_performers_pct
        saved_top_performers_names,
        saved_top_performers,
        saved_combine_top_performers,
        saved_top_performers_pct,
    )

    # Verify the results
    assert len(result) == 16, "Should return 16 values"
    assert result[0] is not None, "Should return a figure"
    assert result[1] == "2013-01-01", "Should return the input start date"
    assert result[2] == "2023-01-01", "Should return the input end date"
    assert result[3] == ["Fund1", "Fund2"], "Should return the input search"
    assert result[4] == "2013-01-01", "Should return the input start date"
    assert result[5] == "2023-01-01", "Should return the input end date"
    assert result[6] == ["Fund1", "Fund2"], "Should return the input search"
    assert result[7] is not None, "Should return a figure"
    assert result[8] is True, "Should return True for loading output"
    assert result[9] == ["Fund3", "Fund4"], "Should return the top performers"
    assert result[10] == "yes", "Should return the input top performers"
    assert result[11] == "no", "Should return the input combine top performers"
    assert result[12] == 15, "Should return the input top performers percentage"
    assert result[13] == "yes", "Should return the input top performers"
    assert result[14] == "no", "Should return the input combine top performers"
    assert result[15] == 15, "Should return the input top performers percentage"

    # Verify that the algo methods were called with the correct parameters
    mock_algo.get_top_performing_assets.assert_called_once_with(
        time_periods=[("2013-01-01", "2023-01-01")],
        top_percent=15 / 100,
    )
    mock_algo.plot_dots.assert_called_once_with(
        start_date="2013-01-01",
        end_date="2023-01-01",
        fund_set=["Fund1", "Fund2"],
        top_performers=["Fund3", "Fund4"],
    )

    # Test with top_performers="no"
    mock_algo.get_top_performing_assets.reset_mock()
    mock_algo.plot_dots.reset_mock()

    result = plot_dots(
        "trigger",  # trigger
        1,  # Click
        "2013-01-01",  # start
        "2023-01-01",  # end
        ["Fund1", "Fund2"],  # search
        saved_start,
        saved_end,
        saved_find_fund,
        saved_figure,
        "no",  # top_performers
        "no",  # combine_top_performers
        15,  # top_performers_pct
        saved_top_performers_names,
        saved_top_performers,
        saved_combine_top_performers,
        saved_top_performers_pct,
    )

    # Verify the results
    assert len(result) == 16, "Should return 16 values"
    assert result[9] == [], "Should return empty list for top performers"

    # Verify that the algo methods were called with the correct parameters
    mock_algo.get_top_performing_assets.assert_not_called()
    mock_algo.plot_dots.assert_called_once_with(
        start_date="2013-01-01",
        end_date="2023-01-01",
        fund_set=["Fund1", "Fund2"],
        top_performers=[],
    )

    # Test with combine_top_performers="yes"
    mock_algo.get_top_performing_assets.reset_mock()
    mock_algo.plot_dots.reset_mock()
    mock_algo.get_top_performing_assets.return_value = ["Fund5", "Fund6", "Fund7"]

    result = plot_dots(
        "trigger",  # trigger
        1,  # Click
        "2013-01-01",  # start
        "2023-01-01",  # end
        ["Fund1", "Fund2"],  # search
        saved_start,
        saved_end,
        saved_find_fund,
        saved_figure,
        "yes",  # top_performers
        "yes",  # combine_top_performers
        15,  # top_performers_pct
        saved_top_performers_names,
        saved_top_performers,
        saved_combine_top_performers,
        saved_top_performers_pct,
    )

    # Verify the results
    assert len(result) == 16, "Should return 16 values"
    assert result[9] == [], "Should return empty list for top performers (no common elements)"

    # Verify that the algo methods were called with the correct parameters
    mock_algo.get_top_performing_assets.assert_called_once_with(
        time_periods=[("2013-01-01", "2023-01-01")],
        top_percent=15 / 100,
    )
    mock_algo.plot_dots.assert_called_once_with(
        start_date="2013-01-01",
        end_date="2023-01-01",
        fund_set=["Fund1", "Fund2"],
        top_performers=[],
    )
