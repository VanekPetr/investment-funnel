"""
Tests for the lifecycle page components and functionality.

This module contains tests for the lifecycle_page.py module, which
provides the components and functionality for the lifecycle investments page
of the investment funnel dashboard.
"""

from unittest.mock import MagicMock, patch

import dash_bootstrap_components as dbc
import pytest
from dash import dcc, html

from funnel.dashboard.components_and_styles.lifecycle_page import (
    _divs,
    _generate_lifecycle_plot,
    create_lifecycle_layout,
    register_callbacks,
)


def test_generate_lifecycle_plot_valid_inputs(algo):
    """
    Test that _generate_lifecycle_plot works correctly with valid inputs.

    This test verifies that the _generate_lifecycle_plot function can generate plots
    with valid inputs without raising any exceptions.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs
    model = "MST"
    model_spec = 2
    pick_top = 2
    scen_model = "Bootstrap"
    scen_spec = 250
    start_data = algo.min_date
    end_train = algo.max_date
    end_year = 2040
    portfolio_value = 100000
    yearly_withdraws = 1000
    risk_preference = 15

    # Call the function
    glidepaths_figure, performance_figure, lifecycle_all_figure, subset_of_assets = _generate_lifecycle_plot(
        algo,
        model=model,
        model_spec=model_spec,
        pick_top=pick_top,
        scen_model=scen_model,
        scen_spec=scen_spec,
        start_data=start_data,
        end_train=end_train,
        end_year=end_year,
        portfolio_value=portfolio_value,
        yearly_withdraws=yearly_withdraws,
        risk_preference=risk_preference,
    )

    # Verify the results
    assert glidepaths_figure is not None, "Glidepaths figure should be created"
    assert performance_figure is not None, "Performance figure should be created"
    assert lifecycle_all_figure is not None, "Lifecycle all figure should be created"
    assert subset_of_assets is not None, "Subset of assets should be created"
    assert isinstance(glidepaths_figure, dcc.Graph), "Glidepaths figure should be a Graph"
    assert isinstance(performance_figure, dcc.Graph), "Performance figure should be a Graph"
    assert isinstance(lifecycle_all_figure, dcc.Graph), "Lifecycle all figure should be a Graph"
    assert isinstance(subset_of_assets, list), "Subset of assets should be a list"


def test_generate_lifecycle_plot_invalid_model(algo):
    """
    Test that _generate_lifecycle_plot raises an error when model is invalid.

    This test verifies that the _generate_lifecycle_plot function raises a ValueError
    when model is not "MST" or "Clustering".

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs with invalid model
    model = "InvalidModel"
    model_spec = 2
    pick_top = 2
    scen_model = "Bootstrap"
    scen_spec = 250
    start_data = algo.min_date
    end_train = algo.max_date
    end_year = 2040
    portfolio_value = 100000
    yearly_withdraws = 1000
    risk_preference = 15

    # Call the function and expect a ValueError
    with pytest.raises(ValueError, match="Invalid model type"):
        _generate_lifecycle_plot(
            algo,
            model=model,
            model_spec=model_spec,
            pick_top=pick_top,
            scen_model=scen_model,
            scen_spec=scen_spec,
            start_data=start_data,
            end_train=end_train,
            end_year=end_year,
            portfolio_value=portfolio_value,
            yearly_withdraws=yearly_withdraws,
            risk_preference=risk_preference,
        )


def test_generate_lifecycle_plot_invalid_scenario_model(algo):
    """
    Test that _generate_lifecycle_plot raises an error when scen_model is invalid.

    This test verifies that the _generate_lifecycle_plot function raises a ValueError
    when scen_model is not "Bootstrap" or "MonteCarlo".

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs with invalid scenario model
    model = "MST"
    model_spec = 2
    pick_top = 2
    scen_model = "InvalidScenModel"
    scen_spec = 250
    start_data = algo.min_date
    end_train = algo.max_date
    end_year = 2040
    portfolio_value = 100000
    yearly_withdraws = 1000
    risk_preference = 15

    # Call the function and expect a ValueError
    with pytest.raises(ValueError):
        _generate_lifecycle_plot(
            algo,
            model=model,
            model_spec=model_spec,
            pick_top=pick_top,
            scen_model=scen_model,
            scen_spec=scen_spec,
            start_data=start_data,
            end_train=end_train,
            end_year=end_year,
            portfolio_value=portfolio_value,
            yearly_withdraws=yearly_withdraws,
            risk_preference=risk_preference,
        )


def test_generate_lifecycle_plot_invalid_numeric_parameters(algo):
    """
    Test that _generate_lifecycle_plot raises an error when numeric parameters are invalid.

    This test verifies that the _generate_lifecycle_plot function raises a ValueError
    when numeric parameters are invalid (e.g., negative or zero).

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs with valid parameters
    model = "MST"
    model_spec = 2
    pick_top = 2
    scen_model = "Bootstrap"
    scen_spec = 250
    start_data = algo.min_date
    end_train = algo.max_date
    end_year = 2040
    portfolio_value = 100000
    yearly_withdraws = 1000
    risk_preference = 15

    # Test with invalid model_spec (zero)
    with pytest.raises(ValueError, match="model_spec must be positive"):
        _generate_lifecycle_plot(
            algo,
            model=model,
            model_spec=0,
            pick_top=pick_top,
            scen_model=scen_model,
            scen_spec=scen_spec,
            start_data=start_data,
            end_train=end_train,
            end_year=end_year,
            portfolio_value=portfolio_value,
            yearly_withdraws=yearly_withdraws,
            risk_preference=risk_preference,
        )

    # Test with invalid pick_top (zero)
    with pytest.raises(ValueError, match="pick_top must be positive"):
        _generate_lifecycle_plot(
            algo,
            model=model,
            model_spec=model_spec,
            pick_top=0,
            scen_model=scen_model,
            scen_spec=scen_spec,
            start_data=start_data,
            end_train=end_train,
            end_year=end_year,
            portfolio_value=portfolio_value,
            yearly_withdraws=yearly_withdraws,
            risk_preference=risk_preference,
        )

    # Test with invalid scen_spec (zero)
    with pytest.raises(ValueError, match="scen_spec must be positive"):
        _generate_lifecycle_plot(
            algo,
            model=model,
            model_spec=model_spec,
            pick_top=pick_top,
            scen_model=scen_model,
            scen_spec=0,
            start_data=start_data,
            end_train=end_train,
            end_year=end_year,
            portfolio_value=portfolio_value,
            yearly_withdraws=yearly_withdraws,
            risk_preference=risk_preference,
        )

    # Test with invalid portfolio_value (zero)
    with pytest.raises(ValueError, match="portfolio_value must be positive"):
        _generate_lifecycle_plot(
            algo,
            model=model,
            model_spec=model_spec,
            pick_top=pick_top,
            scen_model=scen_model,
            scen_spec=scen_spec,
            start_data=start_data,
            end_train=end_train,
            end_year=end_year,
            portfolio_value=0,
            yearly_withdraws=yearly_withdraws,
            risk_preference=risk_preference,
        )

    # Test with invalid yearly_withdraws (negative)
    with pytest.raises(ValueError, match="yearly_withdraws must be non-negative"):
        _generate_lifecycle_plot(
            algo,
            model=model,
            model_spec=model_spec,
            pick_top=pick_top,
            scen_model=scen_model,
            scen_spec=scen_spec,
            start_data=start_data,
            end_train=end_train,
            end_year=end_year,
            portfolio_value=portfolio_value,
            yearly_withdraws=-1,
            risk_preference=risk_preference,
        )

    # Test with invalid risk_preference (zero)
    with pytest.raises(ValueError, match="risk_preference must be between 0 and 100"):
        _generate_lifecycle_plot(
            algo,
            model=model,
            model_spec=model_spec,
            pick_top=pick_top,
            scen_model=scen_model,
            scen_spec=scen_spec,
            start_data=start_data,
            end_train=end_train,
            end_year=end_year,
            portfolio_value=portfolio_value,
            yearly_withdraws=yearly_withdraws,
            risk_preference=0,
        )

    # Test with invalid risk_preference (greater than 100)
    with pytest.raises(ValueError, match="risk_preference must be between 0 and 100"):
        _generate_lifecycle_plot(
            algo,
            model=model,
            model_spec=model_spec,
            pick_top=pick_top,
            scen_model=scen_model,
            scen_spec=scen_spec,
            start_data=start_data,
            end_train=end_train,
            end_year=end_year,
            portfolio_value=portfolio_value,
            yearly_withdraws=yearly_withdraws,
            risk_preference=101,
        )


def test_generate_lifecycle_plot_ml_algorithm_error(algo):
    """
    Test that _generate_lifecycle_plot handles ML algorithm errors correctly.

    This test verifies that the _generate_lifecycle_plot function raises a ValueError
    when the ML algorithm (MST or Clustering) raises an exception.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs
    model = "MST"
    model_spec = 2
    pick_top = 2
    scen_model = "Bootstrap"
    scen_spec = 250
    start_data = algo.min_date
    end_train = algo.max_date
    end_year = 2040
    portfolio_value = 100000
    yearly_withdraws = 1000
    risk_preference = 15

    # Mock the algo.mst method to raise an exception
    with patch.object(algo, 'mst', side_effect=Exception("Test error")):
        with pytest.raises(ValueError, match="Failed to run ML algorithm: Test error"):
            _generate_lifecycle_plot(
                algo,
                model=model,
                model_spec=model_spec,
                pick_top=pick_top,
                scen_model=scen_model,
                scen_spec=scen_spec,
                start_data=start_data,
                end_train=end_train,
                end_year=end_year,
                portfolio_value=portfolio_value,
                yearly_withdraws=yearly_withdraws,
                risk_preference=risk_preference,
            )


def test_generate_lifecycle_plot_empty_subset(algo):
    """
    Test that _generate_lifecycle_plot handles empty subset of assets correctly.

    This test verifies that the _generate_lifecycle_plot function raises a ValueError
    when the ML algorithm returns an empty subset of assets.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs
    model = "MST"
    model_spec = 2
    pick_top = 2
    scen_model = "Bootstrap"
    scen_spec = 250
    start_data = algo.min_date
    end_train = algo.max_date
    end_year = 2040
    portfolio_value = 100000
    yearly_withdraws = 1000
    risk_preference = 15

    # Mock the algo.mst method to return an empty subset
    with patch.object(algo, 'mst', return_value=(None, [])):
        with pytest.raises(ValueError, match="No assets were selected by the ML algorithm"):
            _generate_lifecycle_plot(
                algo,
                model=model,
                model_spec=model_spec,
                pick_top=pick_top,
                scen_model=scen_model,
                scen_spec=scen_spec,
                start_data=start_data,
                end_train=end_train,
                end_year=end_year,
                portfolio_value=portfolio_value,
                yearly_withdraws=yearly_withdraws,
                risk_preference=risk_preference,
            )


def test_generate_lifecycle_plot_lifecycle_error(algo):
    """
    Test that _generate_lifecycle_plot handles lifecycle scenario analysis errors correctly.

    This test verifies that the _generate_lifecycle_plot function raises a ValueError
    when the lifecycle_scenario_analysis method raises an exception.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs
    model = "MST"
    model_spec = 2
    pick_top = 2
    scen_model = "Bootstrap"
    scen_spec = 250
    start_data = algo.min_date
    end_train = algo.max_date
    end_year = 2040
    portfolio_value = 100000
    yearly_withdraws = 1000
    risk_preference = 15

    # Mock the algo.mst method to return a valid subset
    with patch.object(algo, 'mst', return_value=(None, ['asset1', 'asset2'])):
        # Mock the algo.lifecycle_scenario_analysis method to raise an exception
        with patch.object(algo, 'lifecycle_scenario_analysis', side_effect=Exception("Test error")):
            with pytest.raises(ValueError, match="Failed to run lifecycle scenario analysis: Test error"):
                _generate_lifecycle_plot(
                    algo,
                    model=model,
                    model_spec=model_spec,
                    pick_top=pick_top,
                    scen_model=scen_model,
                    scen_spec=scen_spec,
                    start_data=start_data,
                    end_train=end_train,
                    end_year=end_year,
                    portfolio_value=portfolio_value,
                    yearly_withdraws=yearly_withdraws,
                    risk_preference=risk_preference,
                )


def test_generate_lifecycle_plot_gaussian_kde_error(algo):
    """
    Test that _generate_lifecycle_plot handles gaussian_kde errors correctly.

    This test verifies that the _generate_lifecycle_plot function handles the specific
    gaussian_kde dimensionality error correctly, returning placeholder figures with error messages.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs
    model = "MST"
    model_spec = 2
    pick_top = 2
    scen_model = "Bootstrap"
    scen_spec = 250
    start_data = algo.min_date
    end_train = algo.max_date
    end_year = 2040
    portfolio_value = 100000
    yearly_withdraws = 1000
    risk_preference = 15

    # Create a gaussian_kde error message
    kde_error = Exception("The data appears to lie in a lower-dimensional "
                          "subspace of the space in which it is expressed. "
                          "This has resulted in a singular data "
                          "covariance matrix, which cannot be treated "
                          "using the algorithms implemented in `gaussian_kde`.")

    # Mock the algo.mst method to return a valid subset
    with (patch.object(algo, 'mst', return_value=(None, ['asset1', 'asset2']))):
        # Mock the algo.lifecycle_scenario_analysis method to raise a gaussian_kde error
        with patch.object(algo, 'lifecycle_scenario_analysis', side_effect=kde_error):
            # Call the function
            glidepaths_figure, performance_figure, lifecycle_all_figure, subset_of_assets = _generate_lifecycle_plot(
                algo,
                model=model,
                model_spec=model_spec,
                pick_top=pick_top,
                scen_model=scen_model,
                scen_spec=scen_spec,
                start_data=start_data,
                end_train=end_train,
                end_year=end_year,
                portfolio_value=portfolio_value,
                yearly_withdraws=yearly_withdraws,
                risk_preference=risk_preference,
            )

            # Verify the results
            assert glidepaths_figure is not None, "Glidepaths figure should be created"
            assert performance_figure is not None, "Performance figure should be created"
            assert lifecycle_all_figure is not None, "Lifecycle all figure should be created"
            assert subset_of_assets is not None, "Subset of assets should be created"
            assert isinstance(glidepaths_figure, dcc.Graph), "Glidepaths figure should be a Graph"
            assert isinstance(performance_figure, dcc.Graph), "Performance figure should be a Graph"
            assert isinstance(lifecycle_all_figure, dcc.Graph), "Lifecycle all figure should be a Graph"
            assert isinstance(subset_of_assets, list), "Subset of assets should be a list"

            # Check that the figures have error messages
            assert "Error" in str(performance_figure.figure['layout']['title'])
            assert "Error" in str(lifecycle_all_figure.figure['layout']['title'])
            assert "insufficient variation" in str(performance_figure.figure['layout']['annotations'][0]['text'])


def test_create_lifecycle_layout(algo):
    """
    Test that create_lifecycle_layout creates the expected layout.

    This test verifies that the create_lifecycle_layout function creates
    the expected layout for the lifecycle page.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Call the function
    layout = create_lifecycle_layout(algo)

    # Verify the results
    assert layout is not None, "Layout should be created"
    assert isinstance(layout, html.Div), "Layout should be a Div"

    # Check that the layout has the expected structure
    assert len(layout.children) == 1, "Layout should have one child"
    assert isinstance(layout.children[0], dbc.Row), "Layout child should be a Row"


def test_divs_creates_components(algo):
    """
    Test that _divs creates the expected components.

    This test verifies that the _divs function creates the expected
    components for the lifecycle page.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Call the function
    components = _divs(algo)

    # Verify the results
    assert components is not None, "Components should be created"
    assert hasattr(components, 'options'), "Components should have options"
    assert hasattr(components, 'results'), "Components should have results"
    assert hasattr(components, 'spinner'), "Components should have spinner"

    # Check that the components have the expected structure
    assert isinstance(components.options, html.Div), "options should be a Div"
    assert isinstance(components.results, html.Div), "results should be a Div"
    assert isinstance(components.spinner, html.Div), "spinner should be a Div"


def test_register_callbacks(algo):
    """
    Test that register_callbacks registers the expected callbacks.

    This test verifies that the register_callbacks function registers
    the expected callbacks for the lifecycle page.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Create a mock app
    app = MagicMock()

    # Call the function
    register_callbacks(app, algo)

    # Verify that the callbacks were registered
    assert app.callback.call_count == 3, "Four callbacks should be registered"
