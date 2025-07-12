"""
Tests for the AI feature selection page components and functionality.

This module contains tests for the ai_feature_selection_page.py module, which
provides the components and functionality for the AI feature selection page
of the investment funnel dashboard.
"""

from unittest.mock import MagicMock, patch

import dash_bootstrap_components as dbc
import pytest
from dash import dcc, html

from funnel.dashboard.components_and_styles.ai_feature_selection_page import (
    _divs,
    create_ai_feature_selection_layout,
    generate_plot_ml,
    register_callbacks,
)


def test_generate_plot_ml_valid_inputs(algo):
    """
    Test that generate_plot_ml works correctly with valid inputs.

    This test verifies that the generate_plot_ml function can generate a plot
    with valid inputs without raising any exceptions.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs
    model = "MST"
    spec = 2
    start_date = algo.min_date
    end_date = algo.max_date

    # Mock the necessary algo methods to avoid actual computation
    with patch.object(algo, 'mst', return_value=(
        {'data': [], 'layout': {}},  # fig
        ['asset1', 'asset2']  # ai_subset
    )):
        with patch.object(algo, 'get_stat', return_value=MagicMock(
            loc=MagicMock(return_value=MagicMock(
                __getitem__=lambda self, item: MagicMock(
                    round=lambda precision: [1.0, 2.0]
                )
            ))
        )):
            # Call the function
            generated_figure, ai_subset, ai_table_records = generate_plot_ml(
                algo,
                model=model,
                spec=spec,
                start_date=start_date,
                end_date=end_date,
            )

    # Verify the results
    assert generated_figure is not None, "Figure should be created"
    assert ai_subset is not None, "Subset of assets should be created"
    assert ai_table_records is not None, "Table data should be created"
    assert isinstance(generated_figure, dcc.Graph), "Generated figure should be a Graph"
    assert isinstance(ai_subset, list), "Subset of assets should be a list"
    assert isinstance(ai_table_records, dict), "Table data should be a dict"


def test_generate_plot_ml_clustering(algo):
    """
    Test that generate_plot_ml works correctly with Clustering model.

    This test verifies that the generate_plot_ml function can generate a plot
    with the Clustering model without raising any exceptions.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs
    model = "Cluster"
    spec = 3
    start_date = algo.min_date
    end_date = algo.max_date

    # Mock the necessary algo methods to avoid actual computation
    with patch.object(algo, 'clustering', return_value=(
        {'data': [], 'layout': {}},  # fig
        ['asset1', 'asset2', 'asset3']  # ai_subset
    )):
        with patch.object(algo, 'get_stat', return_value=MagicMock(
            loc=MagicMock(return_value=MagicMock(
                __getitem__=lambda self, item: MagicMock(
                    round=lambda precision: [1.0, 2.0, 3.0]
                )
            ))
        )):
            # Call the function
            generated_figure, ai_subset, ai_table_records = generate_plot_ml(
                algo,
                model=model,
                spec=spec,
                start_date=start_date,
                end_date=end_date,
            )

    # Verify the results
    assert generated_figure is not None, "Figure should be created"
    assert ai_subset is not None, "Subset of assets should be created"
    assert ai_table_records is not None, "Table data should be created"
    assert isinstance(generated_figure, dcc.Graph), "Generated figure should be a Graph"
    assert isinstance(ai_subset, list), "Subset of assets should be a list"
    assert isinstance(ai_table_records, dict), "Table data should be a dict"


def test_generate_plot_ml_invalid_model(algo):
    """
    Test that generate_plot_ml raises an error when model is invalid.

    This test verifies that the generate_plot_ml function raises a ValueError
    when model is not "MST" or "Cluster".

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs with invalid model
    model = "InvalidModel"
    spec = 2
    start_date = algo.min_date
    end_date = algo.max_date

    # Call the function and expect a ValueError
    with pytest.raises(ValueError, match="Invalid model type"):
        generate_plot_ml(
            algo,
            model=model,
            spec=spec,
            start_date=start_date,
            end_date=end_date,
        )


def test_generate_plot_ml_invalid_spec(algo):
    """
    Test that generate_plot_ml raises an error when spec is invalid.

    This test verifies that the generate_plot_ml function raises a ValueError
    when spec is not positive.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs with invalid spec
    model = "MST"
    spec = 0
    start_date = algo.min_date
    end_date = algo.max_date

    # Call the function and expect a ValueError
    with pytest.raises(ValueError, match="spec must be positive"):
        generate_plot_ml(
            algo,
            model=model,
            spec=spec,
            start_date=start_date,
            end_date=end_date,
        )


def test_generate_plot_ml_missing_dates(algo):
    """
    Test that generate_plot_ml raises an error when dates are missing.

    This test verifies that the generate_plot_ml function raises a ValueError
    when start_date or end_date is missing.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs with missing start_date
    model = "MST"
    spec = 2
    start_date = None
    end_date = algo.max_date

    # Call the function and expect a ValueError
    with pytest.raises(ValueError, match="All date parameters must be provided"):
        generate_plot_ml(
            algo,
            model=model,
            spec=spec,
            start_date=start_date,
            end_date=end_date,
        )

    # Define test inputs with missing end_date
    start_date = algo.min_date
    end_date = None

    # Call the function and expect a ValueError
    with pytest.raises(ValueError, match="All date parameters must be provided"):
        generate_plot_ml(
            algo,
            model=model,
            spec=spec,
            start_date=start_date,
            end_date=end_date,
        )


def test_generate_plot_ml_mst_error(algo):
    """
    Test that generate_plot_ml handles MST errors correctly.

    This test verifies that the generate_plot_ml function raises a ValueError
    when the MST algorithm raises an exception.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs
    model = "MST"
    spec = 2
    start_date = algo.min_date
    end_date = algo.max_date

    # Mock the algo.mst method to raise an exception
    with patch.object(algo, 'mst', side_effect=Exception("Test error")):
        with pytest.raises(ValueError, match="Failed to run AI feature selection: Test error"):
            generate_plot_ml(
                algo,
                model=model,
                spec=spec,
                start_date=start_date,
                end_date=end_date,
            )


def test_generate_plot_ml_clustering_error(algo):
    """
    Test that generate_plot_ml handles Clustering errors correctly.

    This test verifies that the generate_plot_ml function raises a ValueError
    when the Clustering algorithm raises an exception.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs
    model = "Cluster"
    spec = 3
    start_date = algo.min_date
    end_date = algo.max_date

    # Mock the algo.clustering method to raise an exception
    with patch.object(algo, 'clustering', side_effect=Exception("Test error")):
        with pytest.raises(ValueError, match="Failed to run AI feature selection: Test error"):
            generate_plot_ml(
                algo,
                model=model,
                spec=spec,
                start_date=start_date,
                end_date=end_date,
            )


def test_generate_plot_ml_get_stat_error(algo):
    """
    Test that generate_plot_ml handles get_stat errors correctly.

    This test verifies that the generate_plot_ml function raises a ValueError
    when the get_stat method raises an exception.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Define test inputs
    model = "MST"
    spec = 2
    start_date = algo.min_date
    end_date = algo.max_date

    # Mock the algo.mst method to return valid results
    with patch.object(algo, 'mst', return_value=(
        {'data': [], 'layout': {}},  # fig
        ['asset1', 'asset2']  # ai_subset
    )):
        # Mock the algo.get_stat method to raise an exception
        with patch.object(algo, 'get_stat', side_effect=Exception("Test error")):
            with pytest.raises(ValueError, match="Failed to run AI feature selection: Test error"):
                generate_plot_ml(
                    algo,
                    model=model,
                    spec=spec,
                    start_date=start_date,
                    end_date=end_date,
                )


def test_create_ai_feature_selection_layout(algo):
    """
    Test that create_ai_feature_selection_layout creates the expected layout.

    This test verifies that the create_ai_feature_selection_layout function creates
    the expected layout for the AI feature selection page.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Call the function
    layout = create_ai_feature_selection_layout(algo)

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
    components for the AI feature selection page.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Call the function
    components = _divs(algo)

    # Verify the results
    assert components is not None, "Components should be created"
    assert hasattr(components, 'options'), "Components should have options"
    assert hasattr(components, 'selection'), "Components should have selection"
    assert hasattr(components, 'graph'), "Components should have graph"
    assert hasattr(components, 'spinner'), "Components should have spinner"

    # Check that the components have the expected structure
    assert isinstance(components.options, html.Div), "options should be a Div"
    assert isinstance(components.selection, html.Div), "selection should be a Div"
    assert isinstance(components.graph, html.Div), "graph should be a Div"
    assert isinstance(components.spinner, html.Div), "spinner should be a Div"


def test_register_callbacks(algo):
    """
    Test that register_callbacks registers the expected callbacks.

    This test verifies that the register_callbacks function registers
    the expected callbacks for the AI feature selection page.

    Args:
        algo: The algorithm bot instance from the algo fixture
    """
    # Create a mock app
    app = MagicMock()

    # Call the function
    register_callbacks(app, algo)

    # Verify that the callbacks were registered
    assert app.callback.call_count == 1, "One callback should be registered"

    # Check the callback (plot_ml)
    call_args = app.callback.call_args_list[0][0]
    assert len(call_args) == 0, "Callback should have no positional arguments"
    call_kwargs = app.callback.call_args_list[0][1]
    assert "Output" in str(call_kwargs), "Callback should have Output"
    assert "Input" in str(call_kwargs), "Callback should have Input"
    assert "State" in str(call_kwargs), "Callback should have State"
