import dash_bootstrap_components as dbc
from dash import html
from dashboard.app_components import (
    sideBar,
    optionBacktest,
    graphResults,
    optionML,
    graphML,
    optionGraph,
    graphOverview,
    optionMyPortfolio
)

'''
# ----------------------------------------------------------------------------------------------------------------------
# MY PORTFOLIO
# ----------------------------------------------------------------------------------------------------------------------
'''
page_4_layout = html.Div([
    # Row 1 - body
    dbc.Row([
        # Row 1, Col 1 - navigation bar
        dbc.Col([sideBar]),
        # Row 1, col 2 - text description
        dbc.Col([optionMyPortfolio]),
        # Row 1, Col 3 - table
        dbc.Col([graphResults]),
    ])
])


'''
# ----------------------------------------------------------------------------------------------------------------------
# BACK-TESTING
# ----------------------------------------------------------------------------------------------------------------------
'''
page_3_layout = html.Div([
    # Row 1 - body
    dbc.Row([
        # Row 1, Col 1 - navigation bar
        dbc.Col([sideBar]),
        # Row 1, col 2 - text description
        dbc.Col([optionBacktest]),
        # Row 1, Col 3 - table
        dbc.Col([graphResults]),
    ])
])


'''
# ----------------------------------------------------------------------------------------------------------------------
# AI Feature Selection
# ----------------------------------------------------------------------------------------------------------------------
'''
page_2_layout = html.Div([
    # Row 1 - body
    dbc.Row([
        # Row 1, Col 1 - navigation bar
        dbc.Col([sideBar]),
        # Row 1, col 2 - set-up
        dbc.Col([optionML]),
        # Row 1, Col 3 - table
        dbc.Col([graphML])
    ])
])


'''
# ----------------------------------------------------------------------------------------------------------------------
# MARKET OVERVIEW
# ----------------------------------------------------------------------------------------------------------------------
'''
page_1_layout = html.Div([
    # Row 1 - body
    dbc.Row([
        # Row 1, Col 1 - navigation bar
        dbc.Col([sideBar]),
        # Row 1, col 2 - text description
        dbc.Col([optionGraph]),
        # Row 1, Col 3 - table
        dbc.Col([graphOverview])
    ])
])