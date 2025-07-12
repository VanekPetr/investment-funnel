"""
Tests for the backtest page components and functionality.

This module contains tests for the backtest_page.py module, which
provides the components and functionality for the backtesting page
of the investment funnel dashboard.
"""

from unittest.mock import MagicMock, patch

import dash_bootstrap_components as dbc
import pytest
from dash import dcc, html

from funnel.dashboard.components_and_styles.backtest_page import (
    _divs,
    _generate_backtest,
    create_backtest_layout,
    register_callbacks,
)


def test_generate_backtest_valid_inputs(algo):
    """
    Test that _generate_backtest works correctly with valid inputs.

    This test verifies that the _generate_backtest function can generate plots
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
    benchmark = ["Benchmark1"]
    start_data = algo.min_date
    end_train = algo.max_date
    start_test = algo.min_date
    end_data = algo.max_date
    solver = "ECOS"
    optimization_model = "CVaR"
    lower_bound = 0

    # Mock the necessary algo methods to avoid actual computation
    with patch.object(algo, 'mst', return_value=(None, ['asset1', 'asset2'])):
        with patch.object(algo, 'backtest', return_value=(
            MagicMock(iloc=[MagicMock(to_list=lambda: [1, 2, 3])]),  # opt_table
            MagicMock(iloc=[MagicMock(to_list=lambda: [4, 5, 6])]),  # bench_table
            {'data': [], 'layout': {}},  # fig_performance
            {'data': [], 'layout': {}}   # fig_composition
        )):
            with patch.object(algo, 'plot_dots', return_value={'data': [], 'layout': {}}):
                # Call the function
                perf_figure, comp_figure, universe_figure, subset_of_assets = _generate_backtest(
                    algo,
                    model=model,
                    model_spec=model_spec,
                    pick_top=pick_top,
                    scen_model=scen_model,
                    scen_spec=scen_spec,
                    benchmark=benchmark,
                    start_data=start_data,
                    end_train=end_train,
                    start_test=start_test,
                    end_data=end_data,
                    solver=solver,
                    optimization_model=optimization_model,
                    lower_bound=lower_bound,
                )

    # Verify the results
    assert perf_figure is not None, "Performance figure should be created"
    assert comp_figure is not None, "Composition figure should be created"
    assert universe_figure is not None, "Universe figure should be created"
    assert subset_of_assets is not None, "Subset of assets should be created"
    assert isinstance(perf_figure, dcc.Graph), "Performance figure should be a Graph"
    assert isinstance(comp_figure, dcc.Graph), "Composition figure should be a Graph"
    assert isinstance(universe_figure, dcc.Graph), "Universe figure should be a Graph"
    assert isinstance(subset_of_assets, list), "Subset of assets should be a list"


def test_generate_backtest_invalid_model(algo):
    """
    Test that _generate_backtest raises an error when model is invalid.

    This test verifies that the _generate_backtest function raises a ValueError
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
    benchmark = ["Benchmark1"]
    start_data = algo.min_date
    end_train = algo.max_date
    start_test = algo.min_date
    end_data = algo.max_date
    solver = "ECOS"
    optimization_model = "CVaR"
    lower_bound = 0

    # Call the function and expect a ValueError
    with pytest.raises(ValueError, match="Invalid model type"):
        _generate_backtest(
            algo,
            model=model,
            model_spec=model_spec,
            pick_top=pick_top,
            scen_model=scen_model,
            scen_spec=scen_spec,
            benchmark=benchmark,
            start_data=start_data,
            end_train=end_train,
            start_test=start_test,
            end_data=end_data,
            solver=solver,
            optimization_model=optimization_model,
            lower_bound=lower_bound,
        )


def test_generate_backtest_invalid_scenario_model(algo):
    """
    Test that _generate_backtest raises an error when scen_model is invalid.

    This test verifies that the _generate_backtest function raises a ValueError
    when scen_model is not "Bootstrap", "Bootstrapping", or "MonteCarlo".

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs with invalid scenario model
    model = "MST"
    model_spec = 2
    pick_top = 2
    scen_model = "InvalidScenModel"
    scen_spec = 250
    benchmark = ["Benchmark1"]
    start_data = algo.min_date
    end_train = algo.max_date
    start_test = algo.min_date
    end_data = algo.max_date
    solver = "ECOS"
    optimization_model = "CVaR"
    lower_bound = 0

    # Call the function and expect a ValueError
    with pytest.raises(ValueError):
        _generate_backtest(
            algo,
            model=model,
            model_spec=model_spec,
            pick_top=pick_top,
            scen_model=scen_model,
            scen_spec=scen_spec,
            benchmark=benchmark,
            start_data=start_data,
            end_train=end_train,
            start_test=start_test,
            end_data=end_data,
            solver=solver,
            optimization_model=optimization_model,
            lower_bound=lower_bound,
        )


def test_generate_backtest_missing_dates(algo):
    """
    Test that _generate_backtest raises an error when dates are missing.

    This test verifies that the _generate_backtest function raises a ValueError
    when any of the date parameters are missing.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs with valid parameters
    model = "MST"
    model_spec = 2
    pick_top = 2
    scen_model = "Bootstrap"
    scen_spec = 250
    benchmark = ["Benchmark1"]
    start_data = algo.min_date
    end_train = algo.max_date
    start_test = algo.min_date
    end_data = algo.max_date
    solver = "ECOS"
    optimization_model = "CVaR"
    lower_bound = 0

    # Test with missing start_data
    with pytest.raises(ValueError, match="All date parameters must be provided"):
        _generate_backtest(
            algo,
            model=model,
            model_spec=model_spec,
            pick_top=pick_top,
            scen_model=scen_model,
            scen_spec=scen_spec,
            benchmark=benchmark,
            start_data=None,
            end_train=end_train,
            start_test=start_test,
            end_data=end_data,
            solver=solver,
            optimization_model=optimization_model,
            lower_bound=lower_bound,
        )

    # Test with missing end_train
    with pytest.raises(ValueError, match="All date parameters must be provided"):
        _generate_backtest(
            algo,
            model=model,
            model_spec=model_spec,
            pick_top=pick_top,
            scen_model=scen_model,
            scen_spec=scen_spec,
            benchmark=benchmark,
            start_data=start_data,
            end_train=None,
            start_test=start_test,
            end_data=end_data,
            solver=solver,
            optimization_model=optimization_model,
            lower_bound=lower_bound,
        )

    # Test with missing start_test
    with pytest.raises(ValueError, match="All date parameters must be provided"):
        _generate_backtest(
            algo,
            model=model,
            model_spec=model_spec,
            pick_top=pick_top,
            scen_model=scen_model,
            scen_spec=scen_spec,
            benchmark=benchmark,
            start_data=start_data,
            end_train=end_train,
            start_test=None,
            end_data=end_data,
            solver=solver,
            optimization_model=optimization_model,
            lower_bound=lower_bound,
        )

    # Test with missing end_data
    with pytest.raises(ValueError, match="All date parameters must be provided"):
        _generate_backtest(
            algo,
            model=model,
            model_spec=model_spec,
            pick_top=pick_top,
            scen_model=scen_model,
            scen_spec=scen_spec,
            benchmark=benchmark,
            start_data=start_data,
            end_train=end_train,
            start_test=start_test,
            end_data=None,
            solver=solver,
            optimization_model=optimization_model,
            lower_bound=lower_bound,
        )


def test_generate_backtest_missing_benchmark(algo):
    """
    Test that _generate_backtest raises an error when benchmark is None.

    This test verifies that the _generate_backtest function raises a ValueError
    when the benchmark parameter is None.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs with valid parameters but None benchmark
    model = "MST"
    model_spec = 2
    pick_top = 2
    scen_model = "Bootstrap"
    scen_spec = 250
    benchmark = None
    start_data = algo.min_date
    end_train = algo.max_date
    start_test = algo.min_date
    end_data = algo.max_date
    solver = "ECOS"
    optimization_model = "CVaR"
    lower_bound = 0

    # Call the function and expect a ValueError
    with pytest.raises(ValueError, match="Benchmark parameter cannot be None"):
        _generate_backtest(
            algo,
            model=model,
            model_spec=model_spec,
            pick_top=pick_top,
            scen_model=scen_model,
            scen_spec=scen_spec,
            benchmark=benchmark,
            start_data=start_data,
            end_train=end_train,
            start_test=start_test,
            end_data=end_data,
            solver=solver,
            optimization_model=optimization_model,
            lower_bound=lower_bound,
        )


def test_generate_backtest_invalid_numeric_parameters(algo):
    """
    Test that _generate_backtest raises an error when numeric parameters are invalid.

    This test verifies that the _generate_backtest function raises a ValueError
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
    benchmark = ["Benchmark1"]
    start_data = algo.min_date
    end_train = algo.max_date
    start_test = algo.min_date
    end_data = algo.max_date
    solver = "ECOS"
    optimization_model = "CVaR"
    lower_bound = 0

    # Test with invalid model_spec (zero)
    with pytest.raises(ValueError, match="model_spec must be positive"):
        _generate_backtest(
            algo,
            model=model,
            model_spec=0,
            pick_top=pick_top,
            scen_model=scen_model,
            scen_spec=scen_spec,
            benchmark=benchmark,
            start_data=start_data,
            end_train=end_train,
            start_test=start_test,
            end_data=end_data,
            solver=solver,
            optimization_model=optimization_model,
            lower_bound=lower_bound,
        )

    # Test with invalid pick_top (zero)
    with pytest.raises(ValueError, match="pick_top must be positive"):
        _generate_backtest(
            algo,
            model=model,
            model_spec=model_spec,
            pick_top=0,
            scen_model=scen_model,
            scen_spec=scen_spec,
            benchmark=benchmark,
            start_data=start_data,
            end_train=end_train,
            start_test=start_test,
            end_data=end_data,
            solver=solver,
            optimization_model=optimization_model,
            lower_bound=lower_bound,
        )

    # Test with invalid scen_spec (zero)
    with pytest.raises(ValueError, match="scen_spec must be positive"):
        _generate_backtest(
            algo,
            model=model,
            model_spec=model_spec,
            pick_top=pick_top,
            scen_model=scen_model,
            scen_spec=0,
            benchmark=benchmark,
            start_data=start_data,
            end_train=end_train,
            start_test=start_test,
            end_data=end_data,
            solver=solver,
            optimization_model=optimization_model,
            lower_bound=lower_bound,
        )

    # Test with invalid lower_bound (negative)
    with pytest.raises(ValueError, match="lower_bound must be non-negative"):
        _generate_backtest(
            algo,
            model=model,
            model_spec=model_spec,
            pick_top=pick_top,
            scen_model=scen_model,
            scen_spec=scen_spec,
            benchmark=benchmark,
            start_data=start_data,
            end_train=end_train,
            start_test=start_test,
            end_data=end_data,
            solver=solver,
            optimization_model=optimization_model,
            lower_bound=-1,
        )


def test_generate_backtest_ml_algorithm_error(algo):
    """
    Test that _generate_backtest handles ML algorithm errors correctly.

    This test verifies that the _generate_backtest function raises a ValueError
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
    benchmark = ["Benchmark1"]
    start_data = algo.min_date
    end_train = algo.max_date
    start_test = algo.min_date
    end_data = algo.max_date
    solver = "ECOS"
    optimization_model = "CVaR"
    lower_bound = 0

    # Mock the algo.mst method to raise an exception
    with patch.object(algo, 'mst', side_effect=Exception("Test error")):
        with pytest.raises(ValueError, match="Failed to run ML algorithm: Test error"):
            _generate_backtest(
                algo,
                model=model,
                model_spec=model_spec,
                pick_top=pick_top,
                scen_model=scen_model,
                scen_spec=scen_spec,
                benchmark=benchmark,
                start_data=start_data,
                end_train=end_train,
                start_test=start_test,
                end_data=end_data,
                solver=solver,
                optimization_model=optimization_model,
                lower_bound=lower_bound,
            )


def test_generate_backtest_empty_subset(algo):
    """
    Test that _generate_backtest handles empty subset of assets correctly.

    This test verifies that the _generate_backtest function raises a ValueError
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
    benchmark = ["Benchmark1"]
    start_data = algo.min_date
    end_train = algo.max_date
    start_test = algo.min_date
    end_data = algo.max_date
    solver = "ECOS"
    optimization_model = "CVaR"
    lower_bound = 0

    # Mock the algo.mst method to return an empty subset
    with patch.object(algo, 'mst', return_value=(None, [])):
        with pytest.raises(ValueError, match="No assets were selected by the ML algorithm"):
            _generate_backtest(
                algo,
                model=model,
                model_spec=model_spec,
                pick_top=pick_top,
                scen_model=scen_model,
                scen_spec=scen_spec,
                benchmark=benchmark,
                start_data=start_data,
                end_train=end_train,
                start_test=start_test,
                end_data=end_data,
                solver=solver,
                optimization_model=optimization_model,
                lower_bound=lower_bound,
            )


def test_generate_backtest_backtest_error(algo):
    """
    Test that _generate_backtest handles backtest errors correctly.

    This test verifies that the _generate_backtest function raises a ValueError
    when the backtest method raises an exception.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs
    model = "MST"
    model_spec = 2
    pick_top = 2
    scen_model = "Bootstrap"
    scen_spec = 250
    benchmark = ["Benchmark1"]
    start_data = algo.min_date
    end_train = algo.max_date
    start_test = algo.min_date
    end_data = algo.max_date
    solver = "ECOS"
    optimization_model = "CVaR"
    lower_bound = 0

    # Mock the algo.mst method to return a valid subset
    with patch.object(algo, 'mst', return_value=(None, ['asset1', 'asset2'])):
        # Mock the algo.backtest method to raise an exception
        with patch.object(algo, 'backtest', side_effect=Exception("Test error")):
            with pytest.raises(ValueError, match="Failed to run backtest: Test error"):
                _generate_backtest(
                    algo,
                    model=model,
                    model_spec=model_spec,
                    pick_top=pick_top,
                    scen_model=scen_model,
                    scen_spec=scen_spec,
                    benchmark=benchmark,
                    start_data=start_data,
                    end_train=end_train,
                    start_test=start_test,
                    end_data=end_data,
                    solver=solver,
                    optimization_model=optimization_model,
                    lower_bound=lower_bound,
                )


def test_generate_backtest_datetime_error(algo):
    """
    Test that _generate_backtest handles datetime comparison errors correctly.

    This test verifies that the _generate_backtest function raises a specific ValueError
    when a TypeError with "datetime64[ns, UTC]" and "NoneType" is raised during backtest.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs
    model = "MST"
    model_spec = 2
    pick_top = 2
    scen_model = "Bootstrap"
    scen_spec = 250
    benchmark = ["Benchmark1"]
    start_data = algo.min_date
    end_train = algo.max_date
    start_test = algo.min_date
    end_data = algo.max_date
    solver = "ECOS"
    optimization_model = "CVaR"
    lower_bound = 0

    # Mock the algo.mst method to return a valid subset
    with patch.object(algo, 'mst', return_value=(None, ['asset1', 'asset2'])):
        # Mock the algo.backtest method to raise a TypeError with datetime64 and NoneType
        with patch.object(algo, 'backtest', side_effect=TypeError("Invalid comparison")):
            with pytest.raises(ValueError, match="Failed to run backtest: Invalid comparison"):
                _generate_backtest(
                    algo,
                    model=model,
                    model_spec=model_spec,
                    pick_top=pick_top,
                    scen_model=scen_model,
                    scen_spec=scen_spec,
                    benchmark=benchmark,
                    start_data=start_data,
                    end_train=end_train,
                    start_test=start_test,
                    end_data=end_data,
                    solver=solver,
                    optimization_model=optimization_model,
                    lower_bound=lower_bound,
                )


def test_generate_backtest_plot_dots_error(algo):
    """
    Test that _generate_backtest handles plot_dots errors correctly.

    This test verifies that the _generate_backtest function raises a ValueError
    when the plot_dots method raises an exception.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs
    model = "MST"
    model_spec = 2
    pick_top = 2
    scen_model = "Bootstrap"
    scen_spec = 250
    benchmark = ["Benchmark1"]
    start_data = algo.min_date
    end_train = algo.max_date
    start_test = algo.min_date
    end_data = algo.max_date
    solver = "ECOS"
    optimization_model = "CVaR"
    lower_bound = 0

    # Mock the algo.mst method to return a valid subset
    with patch.object(algo, 'mst', return_value=(None, ['asset1', 'asset2'])):
        # Mock the algo.backtest method to return valid results
        with patch.object(algo, 'backtest', return_value=(
            MagicMock(iloc=[MagicMock(to_list=lambda: [1, 2, 3])]),  # opt_table
            MagicMock(iloc=[MagicMock(to_list=lambda: [4, 5, 6])]),  # bench_table
            {'data': [], 'layout': {}},  # fig_performance
            {'data': [], 'layout': {}}   # fig_composition
        )):
            # Mock the algo.plot_dots method to raise an exception
            with patch.object(algo, 'plot_dots', side_effect=Exception("Test error")):
                with pytest.raises(ValueError, match="Failed to generate universe comparison figure: Test error"):
                    _generate_backtest(
                        algo,
                        model=model,
                        model_spec=model_spec,
                        pick_top=pick_top,
                        scen_model=scen_model,
                        scen_spec=scen_spec,
                        benchmark=benchmark,
                        start_data=start_data,
                        end_train=end_train,
                        start_test=start_test,
                        end_data=end_data,
                        solver=solver,
                        optimization_model=optimization_model,
                        lower_bound=lower_bound,
                    )


def test_create_backtest_layout(algo):
    """
    Test that create_backtest_layout creates the expected layout.

    This test verifies that the create_backtest_layout function creates
    the expected layout for the backtest page.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Call the function
    layout = create_backtest_layout(algo)

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
    components for the backtest page.

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
    the expected callbacks for the backtest page.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Create a mock app
    app = MagicMock()

    # Call the function
    register_callbacks(app, algo)

    # Verify that the callbacks were registered
    assert app.callback.call_count == 4, "Three callbacks should be registered"
