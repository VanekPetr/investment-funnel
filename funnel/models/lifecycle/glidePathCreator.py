import numpy as np
import pandas as pd
import plotly.express as px
from loguru import logger


class RiskCurveGenerator:
    def __init__(self, periods, risk_floor, std_devs):
        self.periods = periods
        self.risk_floor = risk_floor
        self.std_devs = std_devs
        # Ensure floor_starts values are integers
        self.floor_starts = {
            k: int(v)
            for k, v in {
                "Risk Class 3": periods // 2,
                "Risk Class 4": periods // 1.8,
                "Risk Class 5": periods // 1.6,
                "Risk Class 6": periods // 1.3,
                "Risk Class 7": periods // 1.1,
            }.items()
        }
        self.df = pd.DataFrame  # None

    def find_start_value(self, avg_std, floor_start):
        required_area = avg_std * self.periods
        floor_area = self.risk_floor * (self.periods - floor_start)
        curve_area_needed = required_area - floor_area
        start_value = 1.565 * curve_area_needed / floor_start
        return max(start_value, self.risk_floor)

    def generate_curve(self, avg_std, floor_start):
        start_value = self.find_start_value(avg_std, floor_start)
        x = np.arange(1, self.periods + 1)
        curve = np.maximum(start_value * (1 - (x / floor_start) ** 2), self.risk_floor)
        # Explicitly convert floor_start to an integer for slicing
        floor_start = int(floor_start)
        if floor_start < self.periods:
            curve[floor_start - 1 :] = self.risk_floor
        return curve

    def plot_curves(self):
        if self.df is not None:
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

            fig = px.line(
                self.df,
                labels={"value": "Annual Standard Deviation", "index": "Period"},
                title="Concave risk budget glide paths for each Risk Class in the Investment Funnel",
                color_discrete_sequence=colors,  # Use custom colors
            )
            fig.update_layout(
                yaxis_title="Annual Standard Deviation",
                xaxis_title="Period",
                legend_title="Risk Class",
                template="plotly_white",
            )
            fig.show()

            return fig
        else:
            print("No curves generated. Please run generate_curves() first.")

    def generate_curves(self):
        self.df = pd.DataFrame(index=range(self.periods))
        for risk_class, avg_std in self.std_devs.items():
            floor_start = self.floor_starts[risk_class]
            self.df[risk_class] = self.generate_curve(avg_std, floor_start)
        logger.debug(
            f"We have created 5 different concave gliding paths from the Investment Funnel risk classes. "
            f"The average annual Standard Deviation for the gliding paths are: \n {self.df.mean()}"
        )

        glide_path_fig = RiskCurveGenerator.plot_curves(self)

        return self.df, glide_path_fig

    @staticmethod
    def filter_columns_by_risk_class(df, risk_class):
        # Convert risk_class numbers to strings for matching in column names
        risk_class_str = [str(rc) for rc in risk_class]

        # Identify columns that contain any of the specified risk class numbers
        selected_columns = [
            col for col in df.columns if any(rc in col for rc in risk_class_str)
        ]

        # Filter the DataFrame to keep only the selected columns
        filtered_df = df[selected_columns]
        logger.debug(
            f"Qua the chosen risk classes, we only optimize portfolios for {selected_columns}"
        )
        return filtered_df
