import numpy as np
import pandas as pd
import plotly.express as px


def generate_risk_profiles(
    n_periods: int, initial_risk: float, minimum_risk: float
) -> (pd.DataFrame, px.line):
    """Generate risk profiles for glide paths."""

    df = pd.DataFrame(index=range(n_periods))
    x_values = np.linspace(0, 1, n_periods)

    df["Linear GP"] = np.linspace(initial_risk, minimum_risk, n_periods)
    df["Concave GP"] = initial_risk - (initial_risk - minimum_risk) * (x_values**2)
    df["Convex GP"] = initial_risk - (initial_risk - minimum_risk) * np.sqrt(x_values)

    fig = px.line(
        df,
        labels={"value": "Risk", "index": "Period"},
        title="Risk Budget Glide Paths",
    ).update_traces(mode="lines")
    fig.update_layout(
        yaxis_title="Annual Standard Deviation",
        xaxis_title="Period",
        legend_title="Risk Budget Glide Paths",
        height=900,
        width=1500,
        template="plotly_white",
    )

    # Update y-axis properties
    fig.update_yaxes(
        showgrid=True,
        showline=True,
        gridcolor="rgba(211, 211, 211, 0.5)",
        tickformat=".0%",
        title_font=dict(size=15),
        range=[0, initial_risk * 1.1],
    )
    # Update x-axis properties, setting it to reverse
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        title_font=dict(size=15),
        gridcolor="rgba(211, 211, 211, 0.5)",
    )

    # Colors and line width
    colors = [
        "#99A4AE",  # gray50
        "#3b4956",  # dark
        "#b7ada5",  # secondary
        "#4099da",  # blue
        "#8ecdc8",  # aqua
        "#e85757",  # coral
        "#fdd779",  # sun
        "#644c76",  # eggplant
        "#D8D1CA",  # warmGray50
        # Additional colors if needed
    ]

    for i, trace in enumerate(fig.data):
        trace.line.color = colors[i % len(colors)]
        trace.line.width = 2.5

    fig.layout.yaxis.tickformat = ",.1%"

    # fig.show()
    return df, fig
