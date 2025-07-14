"""
Market Overview model module for the Investment Funnel dashboard.

This module provides the data models and processing logic for the Overview page.
It includes functionality for visualizing the investment universe, searching for
specific funds, and highlighting top-performing assets.
"""

from typing import Any

from dash import Output, State, dcc
from pydantic import BaseModel


class OverviewInputs(BaseModel):
    """
    Input model for Market Overview functionality.

    This Pydantic model defines the input parameters required for the market
    overview visualization, including date range, fund search, and top performer options.

    Attributes:
        start_date (str): Start date for the analysis period
        end_date (str): End date for the analysis period
        search (list[str]): List of fund names to highlight in the visualization
        top_performers (str): Whether to highlight top-performing assets ("yes" or "no")
    """
    start_date: str
    end_date: str
    search: list[str] = [""]
    top_performers: str = "no"

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
            State("picker-show", "start_date"),
            State("picker-show", "end_date"),
            State("find-fund", "value"),
            State("top-performers", "value", allow_optional=True),
        ]

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "OverviewInputs":
        """
        Create an OverviewInputs instance from keyword arguments.

        This factory method creates a new instance of the class from
        the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments matching the model's fields

        Returns:
            OverviewInputs: A new instance of the OverviewInputs class
        """
        return cls(**kwargs)

# --- Outputs Model ---
class OverviewOutputs(BaseModel):
    """
    Output model for Market Overview functionality.

    This Pydantic model defines the output data structure for the market
    overview visualization, containing the scatter plot figure of the investment universe.

    Attributes:
        dots_figure (Any): The scatter plot visualization figure (Dash component)
    """
    dots_figure: Any

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
            Output("dotsFig", "children"),
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

def plot_overview(algo, inputs):
    selected_start = str(inputs.start_date)
    selected_end = str(inputs.end_date)

    if inputs.top_performers == "yes":
        top_assets = algo.get_top_performing_assets(
            time_periods=[(selected_start, selected_end)],
            top_percent=10 / 100,
        )
    else:
        top_assets = []

    fig = algo.plot_dots(
        start_date=selected_start,
        end_date=selected_end,
        fund_set=inputs.search,
        top_performers=top_assets,
    )
    generated_figure = dcc.Graph(
        figure=fig,
        style={
            "position": "absolute",
            "right": "0%",
            "bottom": "0%",
            "top": "0%",
            "left": "0%",
        },
    )

    outputs = OverviewOutputs(
        dots_figure=generated_figure,
    )

    return outputs
