from pydantic import BaseModel
from dash import State, Output
from typing import Any
from dash import dcc

class OverviewInputs(BaseModel):
    start_date: str
    end_date: str
    search: list[str] = [""]

    @classmethod
    def as_state_vector(cls) -> list[State]:
        return [
            State("picker-show", "start_date"),
            State("picker-show", "end_date"),
            State("find-fund", "value"),
            State("top-performers", "value", allow_optional=True),
            #State("combine-top-performers", "value", allow_optional=True),
        ]

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "OverviewInputs":
        return cls(**kwargs)

# --- Outputs Model ---
class OverviewOutputs(BaseModel):
    dots_figure: Any

    @classmethod
    def as_output_vector(cls) -> list[Output]:
        return [
            Output("dotsFig", "children"),
        ]

    def as_tuple(self) -> tuple[Any]:
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
