from typing import Any

from dash import Output, State, dcc
from pydantic import BaseModel


class LifecycleInputs(BaseModel):
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
        return cls(**kwargs)

# --- Outputs Model ---
class LifecycleOutputs(BaseModel):
    glidepaths_output_fig: Any
    performance_output_fig: Any
    lifecycle_all_output_fig: Any


    @classmethod
    def as_output_vector(cls) -> list[Output]:
        return [
            Output("glidepaths-output-fig", "children"),
            Output("performance-output-fig", "children"),
            Output("lifecycle-all-output-fig", "children"),
        ]

    def as_tuple(self) -> tuple[Any]:
        return tuple(self.__dict__.values())

def plot_lifecycle(algo, inputs):
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
