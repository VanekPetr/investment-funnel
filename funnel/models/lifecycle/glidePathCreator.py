import pandas as pd
import numpy as np
import plotly.express as px


def generate_risk_profiles(n_periods, initial_risk, minimum_risk):
    # Create a dataframe with index as range of periods
    df = pd.DataFrame(index=range(n_periods))
    x_values = np.linspace(0, 1, n_periods)
    x_values_early = np.linspace(0, 1, int(np.floor(n_periods*0.8)))


    upper_limit_80 = initial_risk - (minimum_risk * 2)
    lower_limit_20 = minimum_risk * 2
    upper_limit_60 = initial_risk - (minimum_risk * 4)
    lower_limit_40 = minimum_risk * 4



    '''
    
    upper_limit_narrow = initial_risk * 0.8
    lower_limit_half = minimum_risk + (initial_risk * 0.5)

    
    # Original curves
    df['Concave risk reduction'] = initial_risk - (initial_risk - minimum_risk) * (x_values ** 2)
    #df['Concave risk reduction narrow'] = initial_risk - (initial_risk - lower_limit_narrow) * (x_values ** 2)
    #df['Convex risk early'] = np.concatenate((df['Convex risk reduction'][:n_periods-5], np.full(5, minimum_risk)))

    # Narrow range curve
    df['Concave risk reduction narrow'] = upper_limit_narrow - (upper_limit_narrow - lower_limit_narrow) * (
                x_values ** 2)

    # Constant risk
    df['Constant_100'] = np.full(n_periods, initial_risk)
    df['Constant_75'] = np.concatenate(
        (np.full(round(n_periods * 0.75), initial_risk), np.full(n_periods - round(n_periods * 0.75), lower_limit_half)))
    df['Constant_50'] = np.concatenate(
        (np.full(round(n_periods * 0.5), initial_risk), np.full(n_periods - round(n_periods * 0.5), lower_limit_half)))
    df['Constant_100_half'] = np.full(n_periods, initial_risk * 0.5)

    # Reversed curves
    df['Concave risk reduction reverse'] = df['Concave risk reduction'][::-1].values
    df['Concave risk reduction narrow reverse'] = df['Concave risk reduction narrow'][::-1].values
    df['Constant_75 reversed'] = df['Constant_75'][::-1].values
    df['Constant_50 reversed'] = df['Constant_50'][::-1].values

    # Adjusted for early completion and constant thereafter
    early_stop = n_periods - 5
    df['Concave risk early'] = np.concatenate(
        (df['Concave risk reduction'][:early_stop], np.full(5, df['Concave risk reduction'][early_stop - 1])))
    '''
    # Other
    #df['Constant risk low'] = np.full(n_periods, initial_risk * 1/2)
    #df['Constant risk high'] = np.full(n_periods, initial_risk * 2/3)

    df['100-0'] = np.linspace(initial_risk, minimum_risk, n_periods)
    df['80-20'] = np.linspace(upper_limit_80, lower_limit_20, n_periods)
    df['60-40'] = np.linspace(upper_limit_60, lower_limit_40, n_periods)

    #df['Concave risk reduction'] = initial_risk - (initial_risk - minimum_risk) * (x_values ** 2)
    #df['Concave risk early'] = np.concatenate((initial_risk - (initial_risk - minimum_risk) * (x_values_early ** 2),np.full(n_periods-len(x_values_early), minimum_risk)))

    #df['Convex risk reduction'] = initial_risk - (initial_risk - minimum_risk) * np.sqrt(x_values)
    #df['Convex risk reduction narrow'] = initial_risk - (initial_risk - lower_limit_narrow) * np.sqrt(x_values)
    #df['Constant_50'] = np.concatenate((np.full(round(n_periods * 0.5), initial_risk*3/4), np.full(n_periods - round(n_periods * 0.5), lower_limit_half)))
    # Reversed curves
    #df['Concave risk reduction reverse'] = df['Concave risk reduction'][::-1].values
    #df['Constant_50 reversed'] = df['Constant_50'][::-1].values
    #df['Convex risk reduction reverse'] = df['Convex risk reduction'][::-1].values
    df['100-0 reverse'] = df['100-0'][::-1].values
    df['80-20 reverse'] = df['80-20'][::-1].values
    df['60-40 reverse'] = df['60-40'][::-1].values

    fig = px.line(
        df,
        labels={"value": "Risk", "index": "Period"},
        title="Risk Budget Glide Paths",
    ).update_traces(mode='lines')
    fig.update_layout(
        yaxis_title="Annual Standard Deviation",
        xaxis_title="Period",
        legend_title="Risk Budget Glide Paths",
        height=900,
        width=1500,
        template="plotly_white",
    )

    # Update y-axis properties
    fig.update_yaxes(showgrid=True, showline=True, gridcolor='rgba(211, 211, 211, 0.5)',
                     tickformat=".0%", title_font=dict(size=15),
                     range=[0, initial_risk * 1.1])
    # Update x-axis properties, setting it to reverse
    fig.update_xaxes(showgrid=False, showline=True, title_font=dict(size=15),
                     gridcolor='rgba(211, 211, 211, 0.5)')

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

    fig.show()
    return df, fig

