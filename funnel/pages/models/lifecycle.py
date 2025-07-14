"""
Lifecycle Investment model module for the Investment Funnel dashboard.

This module provides the data models and processing logic for the Lifecycle page.
It includes functionality for analyzing lifecycle investment strategies with
different risk preferences, time horizons, and withdrawal patterns.
"""

from typing import Any

from dash import Output, State, dcc
from pydantic import BaseModel


class LifecycleInputs(BaseModel):
    """
    Input model for Lifecycle Investment functionality.

    This Pydantic model defines the input parameters required for the lifecycle
    investment analysis, including model selection, date ranges, and investment parameters.

    Attributes:
        model (str): The feature selection model type (e.g., "MST" or "Clustering")
        model_spec (int): The number of clusters or MST runs to perform
        pick_top (float): Number of top-performing assets to select from each cluster
        scen_model (str): The scenario generation model (e.g., "Bootstrap" or "MonteCarlo")
        scen_spec (float): The number of scenarios to generate
        start_date (str): Start date for the analysis period
        end_date (str): End date for the analysis period
        end_year (int): Target end year for the lifecycle investment
        portfolio_value (float): Initial portfolio value
        yearly_withdraws (float): Annual withdrawal amount
        risk_preference (float): Initial risk appetite as a percentage
    """
    model: str
    model_spec: int
    pick_top: float
    scen_model: str
    scen_spec: float
    start_date: str
    end_date: str
    end_year: int
    portfolio_value: float
    yearly_withdraws: float
    risk_preference: float

    @classmethod
    def as_state_vector(cls) -> list[State]:
        """
        Create a list of Dash State objects for callback input capturing.

        This method maps the model's attributes to the corresponding UI components
        to capture their values in a Dash callback.

        Returns:
            list[State]: A list of Dash State objects for the callback
        """
        return [
            State("select-ml-lifecycle", "value"),
            State("slider-lifecycle-ml", "value"),
            State("slider-lifecycle", "value"),
            State("select-scenarios-lifecycle", "value"),
            State("my-slider-2-lifecycle", "value"),
            State("picker-lifecycle", "start_date"),
            State("picker-lifecycle", "end_date"),
            State("slider-final-year-lifecycle", "value"),
            State("initial-portfolio-value-lifecycle", "value"),
            State("yearly-withdraws-lifecycle", "value"),
            State("initial-risk-appetite-lifecycle", "value"),
        ]

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "LifecycleInputs":
        """
        Create a LifecycleInputs instance from keyword arguments.

        This factory method creates a new instance of the class from
        the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments matching the model's fields

        Returns:
            LifecycleInputs: A new instance of the LifecycleInputs class
        """
        return cls(**kwargs)

# --- Outputs Model ---
class LifecycleOutputs(BaseModel):
    """
    Output model for Lifecycle Investment functionality.

    This Pydantic model defines the output data structure for the lifecycle
    investment analysis, including glidepath visualization, performance analysis,
    and portfolio composition over time.

    Attributes:
        glidepaths_output_fig (Any): The glidepath visualization figure (Dash component)
        performance_output_fig (Any): The performance analysis figure (Dash component)
        lifecycle_all_output_fig (Any): The portfolio composition over time figure (Dash component)
    """
    glidepaths_output_fig: Any
    performance_output_fig: Any
    lifecycle_all_output_fig: Any

    @classmethod
    def as_output_vector(cls) -> list[Output]:
        """
        Create a list of Dash Output objects for callback output mapping.

        This method maps the model's attributes to the corresponding UI components
        that will display the results in the Dash application.

        Returns:
            list[Output]: A list of Dash Output objects for the callback
        """
        return [
            Output("glidepaths-output-fig", "children"),
            Output("performance-output-fig", "children"),
            Output("lifecycle-all-output-fig", "children"),
        ]

    def as_tuple(self) -> tuple[Any]:
        """
        Convert the model instance to a tuple of values.

        This method is used to prepare the model's data for returning
        from a Dash callback function.

        Returns:
            tuple[Any]: A tuple containing all the model's attribute values
        """
        return tuple(self.__dict__.values())

def plot_lifecycle(algo, inputs):
    """
    Perform lifecycle investment analysis and generate visualization outputs.

    This function processes the input parameters to run a complete lifecycle investment analysis:
    1. Selects assets using the specified feature selection method (MST or Clustering)
    2. Runs the lifecycle scenario analysis with the specified parameters
    3. Generates glidepath, performance, and portfolio composition visualizations

    Args:
        algo: The algorithm object containing the lifecycle analysis methods
        inputs (LifecycleInputs): The input parameters for the lifecycle analysis

    Returns:
        LifecycleOutputs: The output data containing the visualization figures
    """
    # inputs = LifecycleInputs.from_kwargs(**kwargs)
    # Lifecycle analysis

    # RUN ML algo
    if inputs.model == "MST":
        _, subset_of_assets = algo.mst(
            start_date=inputs.start_date, end_date=inputs.end_date, n_mst_runs=int(inputs.model_spec)
        )
    else:
        _, subset_of_assets = algo.clustering(
            start_date=inputs.start_date,
            end_date=inputs.end_date,
            n_clusters=int(inputs.model_spec),
            n_assets=int(inputs.pick_top),
        )
    # RUN THE LIFECYCLE FUNCTION
    _, _, fig_performance, fig_glidepaths, _, _, fig_composition_all = (
        algo.lifecycle_scenario_analysis(
            subset_of_assets=subset_of_assets,
            scenarios_type=inputs.scen_model,
            n_simulations=int(inputs.scen_spec),
            end_year=inputs.end_year,
            withdrawals=inputs.yearly_withdraws,
            initial_risk_appetite=inputs.risk_preference / 100,
            initial_budget=inputs.portfolio_value,
        )
    )

    performance_figure = dcc.Graph(
        figure=fig_performance, style={"margin": "0%", "height": "800px"}
    )
    glidepaths_figure = dcc.Graph(
        figure=fig_glidepaths, style={"margin": "0%", "height": "800px"}
    )
    lifecycle_all_figure = dcc.Graph(
        figure=fig_composition_all, style={"margin": "0%", "height": "1300px"}
    )

    # Your logic here; placeholder values shown
    outputs = LifecycleOutputs(
        glidepaths_output_fig=glidepaths_figure,
        performance_output_fig=performance_figure,
        lifecycle_all_output_fig=lifecycle_all_figure,
    )

    return outputs
