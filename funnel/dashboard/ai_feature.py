from typing import Any

from dash import Output, State, dcc
from pydantic import BaseModel


class FeatureInput(BaseModel):
    model: str
    spec: int
    start_date: str
    end_date: str

    @classmethod
    def as_state_vector(cls) -> list[State]:
        return [
            State("model-dropdown", "value"),
            State("ML-num-dropdown", "value"),
            State("picker-AI", "start_date"),
            State("picker-AI", "end_date"),
        ]

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "FeatureInput":
        return cls(**kwargs)

# --- Outputs Model ---
class FeatureOutput(BaseModel):
    ml_figure: Any
    ai_result: Any
    ai_number: str

    @classmethod
    def as_output_vector(cls) -> list[Output]:
        return [
            Output("mlFig", "children"),
            Output("AIResult", "data"),
            Output("AInumber", "children"),
        ]

    def as_tuple(self) -> tuple[Any]:
        return tuple(self.__dict__.values())

def plot_ml(algo, inputs):
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
