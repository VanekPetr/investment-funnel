"""
AI Feature Selection model module for the Investment Funnel dashboard.

This module provides the data models and processing logic for the AI Feature Selection page.
It includes functionality for Minimum Spanning Tree (MST) and Clustering methods to reduce
the number of assets in the investment universe.
"""

from typing import Any

from dash import Output, State, dcc
from pydantic import BaseModel


class FeatureInput(BaseModel):
    """
    Input model for AI Feature Selection functionality.

    This Pydantic model defines the input parameters required for the AI feature
    selection process, including the model type, specifications, and date range.

    Attributes:
        model (str): The AI model type to use (e.g., "MST" or "Cluster")
        spec (int): The number of clusters or MST runs to perform
        start_date (str): The start date for the analysis period
        end_date (str): The end date for the analysis period
    """
    model: str
    spec: int
    start_date: str
    end_date: str

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
            State("model-dropdown", "value"),
            State("ML-num-dropdown", "value"),
            State("picker-AI", "start_date"),
            State("picker-AI", "end_date"),
        ]

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "FeatureInput":
        """
        Create a FeatureInput instance from keyword arguments.

        This factory method creates a new instance of the class from
        the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments matching the model's fields

        Returns:
            FeatureInput: A new instance of the FeatureInput class
        """
        return cls(**kwargs)

# --- Outputs Model ---
class FeatureOutput(BaseModel):
    """
    Output model for AI Feature Selection functionality.

    This Pydantic model defines the output data structure for the AI feature
    selection process, including the visualization figure, selected assets data,
    and summary information.

    Attributes:
        ml_figure (Any): The visualization figure (Dash component)
        ai_result (Any): The selected assets data for the data table
        ai_number (str): A text summary of the number of selected assets
    """
    ml_figure: Any
    ai_result: Any
    ai_number: str

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
            Output("mlFig", "children"),
            Output("AIResult", "data"),
            Output("AInumber", "children"),
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

def plot_ml(algo, inputs):
    """
    Perform AI feature selection and generate visualization outputs.

    This function processes the input parameters to run either the Minimum Spanning Tree
    or Clustering algorithm for feature selection. It generates a visualization figure,
    prepares the selected assets data for display in a table, and creates a summary text.

    Args:
        algo: The algorithm object containing the feature selection methods
        inputs (FeatureInput): The input parameters for the feature selection process

    Returns:
        FeatureOutput: The output data containing the visualization and selected assets
    """
    selected_start = str(inputs.start_date)
    selected_end = str(inputs.end_date)

    # MST
    if inputs.model == "MST":
        # RUN THE MINIMUM SPANNING TREE METHOD
        fig, ai_subset = algo.mst(
            start_date=selected_start,
            end_date=selected_end,
            n_mst_runs=inputs.spec,
            plot=True,
        )
        generated_figure = dcc.Graph(
            figure=fig, style={"height": "800px", "margin": "0%"}
        )
    # CLUSTERING
    else:
        fig, ai_subset = algo.clustering(
            start_date=selected_start,
            end_date=selected_end,
            n_clusters=inputs.spec,
            n_assets=10,
            plot=True,
        )
        generated_figure = dcc.Graph(
            figure=fig, style={"height": "800px", "margin": "0%"}
        )

    ai_data = algo.get_stat(start_date=selected_start, end_date=selected_end)
    ai_table = ai_data.loc[
        list(ai_subset),
        [
            "Name",
            "ISIN",
            "Sharpe Ratio",
            "Average Annual Returns",
            "Standard Deviation of Returns",
        ],
    ]
    # ROUNDING
    ai_table["Standard Deviation of Returns"] = round(
        ai_table["Standard Deviation of Returns"], 2
    )
    ai_table["Average Annual Returns"] = round(
        ai_table["Average Annual Returns"], 2
    )

    ml_text = "Number of selected assets: " + str(len(ai_table))

    outputs = FeatureOutput(
        ml_figure=generated_figure,
        ai_result=ai_table.to_dict("records"),
        ai_number=ml_text,
    )

    return outputs
