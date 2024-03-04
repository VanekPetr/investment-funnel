import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots








# Colors for each category
colors = [
    "#4099da",  # blue
    "#b7ada5",  # secondary
    "#8ecdc8",  # aqua
    "#e85757",  # coral
    "#fdd779",  # sun
    "#644c76",  # eggplant
    "#3b4956",  # dark
    "#99A4AE",  # gray50
    "#3B4956",  # gray100
    "#D8D1CA",  # warmGray50
    "#B7ADA5",  # warmGray100
    "#FFFFFF",  # white
]

# Categories for the x-axis
categories = ["Blue", "Secondary", "Aqua", "Coral", "Sun", "Eggplant", "Dark", "Gray50", "Gray100", "WarmGray50", "WarmGray100", "White"]

# Create the bar chart
fig = go.Figure(data=[go.Bar(
    x=categories,
    y=[1] * len(categories),  # Dummy values; replace with your actual data
    marker_color=colors  # Set the bar colors
)])

# Update layout for a cleaner look
fig.update_layout(
    title="Bar Chart with Custom Colors",
    xaxis_title="Category",
    yaxis_title="Value",
    plot_bgcolor='white',
    yaxis=dict(showgrid=False),  # Hide the y-axis grid lines
)

# Show the figure
fig.show()