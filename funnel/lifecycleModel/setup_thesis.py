import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TDF_data = pd.read_excel('C:/Users/MIBEM/OneDrive - Ørsted/Desktop/Thesis/saved_data/dataForAnnualisedTDFvols.xlsx',
                         sheet_name='data_overview')
ann_vols = pd.read_excel('C:/Users/MIBEM/OneDrive - Ørsted/Desktop/Thesis/saved_data/dataForAnnualisedTDFvols.xlsx',
                         sheet_name='campbell')
vanguard = pd.DataFrame({
    'Year': TDF_data['Year'],
    'Stock allocation': TDF_data['Vanguard Stock'],
    'Bond allocation': TDF_data['Vanguard Bond'],
    'Cash allocation': TDF_data['Vanguard Cash'],
})
troweprice = pd.DataFrame({
    'Year': TDF_data['Year'],
    'Stock allocation': TDF_data['TRowePrice Stock'],
    'Bond allocation': TDF_data['TRowePrice Bond'],
    'Cash allocation': TDF_data['TRowePrice Cash'],
})
fidelity = pd.DataFrame({
    'Year': TDF_data['Year'],
    'Stock allocation': TDF_data['Fidelity Stock'],
    'Bond allocation': TDF_data['Fidelity Bond'],
    'Cash allocation': TDF_data['Fidelity Cash'],
})
tdfs = [vanguard, troweprice, fidelity]

colors = [
    "#99A4AE",  # gray50
    "#3b4956",  # dark
    "#b7ada5",  # secondary
]

fig = make_subplots(rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.03)  # Reduced spacing


def add_traces(df, row, colors):
    for i, col in enumerate(["Stock allocation", "Bond allocation", "Cash allocation"]):
        fig.add_trace(
            go.Scatter(
                x=df["Year"],
                y=df[col],
                mode='lines',
                line=dict(width=0.5, color=colors[i]),
                stackgroup='one',
                name=col,
                legendgroup=col,
                showlegend=True if row == 1 else False,
                opacity=1  # Ensure full opacity for the colors
            ),
            row=row, col=1
        )


for i, tdf in enumerate(tdfs, start=1):
    add_traces(tdf, i, colors)

# Update axes properties individually
for i in range(1, 4):
    fig.update_yaxes(title_text="Allocation Percentage" if i == 2 else "",
                     showgrid=True,
                     showline=True,
                     gridcolor='rgba(211, 211, 211, 0.5)', row=i, col=1,
                     tickformat=".0%",  # Formats tick labels as percentages
                     title_font=dict(size=14),  # Adjust the font size as needed
                     )
    fig.update_xaxes(title_text="Years to target date" if i == 3 else "",
                     showgrid=True,
                     showline=True,
                     gridcolor='rgba(211, 211, 211, 0.5)',
                     row=i,
                     col=1,
                     title_font=dict(size=14)  # Adjust the font size as needed
                     )

# Add subplot labels as annotations
subplot_labels = ['a)', 'b)', 'c)']
for i, label in enumerate(subplot_labels, start=1):
    fig.add_annotation(
        text=label,  # Text for the label
        xref="paper", x=-0.1,  # Positioning the label to the left of the plot
        yref=f"y{i}",  # Adjusting yref to target the correct subplot
        y=1,  # Centering the label vertically in the subplot
        showarrow=False,  # Not showing an arrow
        align="right",  # Right align text for consistent positioning
        yanchor="middle",  # Anchor annotation in the middle vertically
        font=dict(  # Set font attributes here
            size=14,  # Adjust the font size as needed
            family="Arial Black"  # Example of a bolder font, adjust based on availability
        )
    )
fig.update_layout(
    legend=dict(
        orientation="h",  # Horizontal orientation
        x=0.5,  # Centered horizontally
        xanchor="center",  # Anchor the legend at its center horizontally
        y=-0.069,  # Position below the x-axis of the bottom plot; adjust as necessary
        yanchor="top",  # Anchor the legend at its top to avoid overlapping with plot area
        font=dict(size=15),  # Adjust font size as needed
    ),
    height=1000, width=850, plot_bgcolor='white', title=None
)
fig.show()

# -------------------------------------------------------------------------
# ------------------------ Annualised Volatilities ------------------------
# -------------------------------------------------------------------------


df = pd.DataFrame({
    'Year': ann_vols['Year'],
    'Stock/Bond correlation': ann_vols['Corr(S,B)'],
    'Timevariant annualised stock vol': ann_vols['Campbell Stocks'],
    'Timevariant annualised bond vol': ann_vols['Campbell Bonds'],
})

# Create a figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces for stock and bond vol on the primary y-axis
fig.add_trace(
    go.Scatter(x=df['Year'], y=df['Timevariant annualised stock vol'], name='Stock Vol', mode='lines',
               line=dict(width=2.5, color=colors[0])),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=df['Year'], y=df['Timevariant annualised bond vol'], name='Bond Vol', mode='lines',
               line=dict(width=2.5, color=colors[1])),
    secondary_y=False,
)

# Add trace for stock/bond correlation on the secondary y-axis
fig.add_trace(
    go.Scatter(x=df['Year'], y=df['Stock/Bond correlation'], name='Stock/Bond Correlation', mode='lines',
               line=dict(width=2.5, color=colors[2])),
    secondary_y=True,
)

# Update axes properties
fig.update_xaxes(title_text="Years to target date", showgrid=False, showline=True, title_font=dict(size=15),
                 gridcolor='rgba(211, 211, 211, 0.5)')

# Update axes properties with y-axis starting at 0 for the primary y-axis
fig.update_yaxes(title_text="Volatility", showgrid=True, showline=True, gridcolor='rgba(211, 211, 211, 0.5)',
                 secondary_y=False, tickformat=".0%", title_font=dict(size=15), range=[0, max(
        df['Timevariant annualised stock vol'].max(), df['Timevariant annualised bond vol'].max()) * 1.1])

# You might want to separately specify the range for the secondary y-axis if it should also start at 0
fig.update_yaxes(title_text="Correlation", showgrid=False, showline=True, secondary_y=True, title_font=dict(size=15),
                 tickformat=".2f", range=[0, df['Stock/Bond correlation'].max() * 1.1])

# Adjusting the legend to be at the bottom center in one row
fig.update_layout(
    legend=dict(
        orientation="h",
        x=0.5,
        xanchor="center",
        y=-0.2,
        yanchor="top",
        font=dict(size=16),
    ),
    height=600,
    width=1200,
    plot_bgcolor='white',
)

fig.show()

# -------------------------------------------------------------------------
# ------------------------ RISK PROFILES ------------------------
# -------------------------------------------------------------------------


df_riskprofiles = pd.DataFrame({
    'Year': TDF_data['Year'],
    'Vanguard': TDF_data['Vanguard ann. Vol'],
    'Fidelity': TDF_data['Fidelity ann. Vol'],
    'T.RowePrice': TDF_data['T.RowePrice ann. Vol'],
})

# Create a figure with secondary y-axis
fig = make_subplots()

# Add traces for stock and bond vol on the primary y-axis
fig.add_trace(
    go.Scatter(x=df['Year'], y=df_riskprofiles['Vanguard'], name='Vanguard Risk Profile', mode='lines',
               line=dict(width=2.5, color=colors[0]))
)

fig.add_trace(
    go.Scatter(x=df['Year'], y=df_riskprofiles['Fidelity'], name='Fidelity Risk Profile', mode='lines',
               line=dict(width=2.5, color=colors[1]))
)

fig.add_trace(
    go.Scatter(x=df['Year'], y=df_riskprofiles['T.RowePrice'], name='T.RowePrice Risk Profile', mode='lines',
               line=dict(width=2.5, color=colors[2]))
)

# Update axes properties
fig.update_xaxes(title_text="Years to target date", showgrid=False, showline=True, title_font=dict(size=15),
                 gridcolor='rgba(211, 211, 211, 0.5)')

# Update axes properties with y-axis starting at 0 for the primary y-axis
fig.update_yaxes(title_text="Volatility p.a.", showgrid=True,
                 showline=True, gridcolor='rgba(211, 211, 211, 0.5)',
                 tickformat=".0%", title_font=dict(size=15),
                 range=[0.06, df_riskprofiles.iloc[:, 1:].values.max() * 1.1])

# Adjusting the legend to be at the bottom center in one row
fig.update_layout(
    legend=dict(
        orientation="h",
        x=0.5,
        xanchor="center",
        y=-0.2,
        yanchor="top",
        font=dict(size=16),
    ),
    height=600,
    width=1200,
    plot_bgcolor='white',
)

fig.show()


# -------------------------------------------------------------------------
# ------------------------ TARGET VOL ------------------------
# -------------------------------------------------------------------------


df_riskprofiles = pd.DataFrame({
    'Year': np.linspace(50, 0, 51),
    'Risk Budget': np.linspace(0.12, 0.04, 51),
})

fig = make_subplots()

fig.add_trace(
    go.Scatter(x=df_riskprofiles['Year'], y=df_riskprofiles['Risk Budget'], name='Risk Budget for target date fund', mode='lines',
               line=dict(width=2.5, color=colors[0]))
)

# Update x-axis properties, setting it to reverse
fig.update_xaxes(title_text="Years to target date", showgrid=False, showline=True, title_font=dict(size=15),
                 gridcolor='rgba(211, 211, 211, 0.5)', autorange="reversed")

# Update y-axis properties
fig.update_yaxes(title_text="Volatility p.a.", showgrid=True, showline=True, gridcolor='rgba(211, 211, 211, 0.5)',
                 tickformat=".0%", title_font=dict(size=15), range=[0.01, df_riskprofiles['Risk Budget'].max() * 1.2])

# Adjust the legend and layout
fig.update_layout(
    legend=dict(
        orientation="h",
        x=0.5,
        xanchor="center",
        y=-0.2,
        yanchor="top",
        font=dict(size=16),
    ),
    height=600,
    width=1200,
    plot_bgcolor='white',
)

fig.show()



# -------------------------------------------------------------------------
# ------------------------ GET COLORS ------------------------
# -------------------------------------------------------------------------
# Define colors
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
]

# Create a figure
fig = go.Figure()

# Add bars
for i, color in enumerate(colors):
    fig.add_trace(go.Bar(
        x=[color],  # Use the color as the label
        y=[1],  # All bars will have the same height
        marker_color=color,  # Set the color of the bar
        name=color,  # Set the name of the bar to its color
        showlegend=False  # Hide legend to keep the plot clean
    ))

# Update layout
fig.update_layout(
    title="Color Palette",
    xaxis_title="Colors",
    yaxis_title="Frequency",
    yaxis=dict(showticklabels=False),  # Hide y-axis ticks
    xaxis=dict(tickangle=45),  # Rotate labels for better readability
    plot_bgcolor="white"  # Set background color to white for contrast
)

# Show figure
fig.show()