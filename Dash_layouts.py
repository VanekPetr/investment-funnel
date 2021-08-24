from Dash_components import *

'''
# ----------------------------------------------------------------------------------------------------------------------
# BACK-TESTING
# ----------------------------------------------------------------------------------------------------------------------
'''
page_3_layout = html.Div([
    # Row 1 - top bar
    # dbc.Row([topBar]),
    # Row 2 - body
    dbc.Row([
        # Row 2, Col 1 - navigation bar
        dbc.Col([sideBar]),
        # Row 2, col 2 - text description
        dbc.Col([optionBacktest]),
        # Row 2, Col 3 - table
        dbc.Col([graphResults]),
    ])
])


'''
# ----------------------------------------------------------------------------------------------------------------------
# AI Feature Selection
# ----------------------------------------------------------------------------------------------------------------------
'''
page_2_layout = html.Div([
    # Row 1 - top bar
    # dbc.Row([topBar]),
    # Row 2 - body
    dbc.Row([
        # Row 2, Col 1 - navigation bar
        dbc.Col([sideBar]),
        # Row 2, col 2 - set-up
        dbc.Col([optionML]),
        # Row 2, Col 3 - table
        dbc.Col([graphML])
    ])
])


'''
# ----------------------------------------------------------------------------------------------------------------------
# MARKET OVERVIEW
# ----------------------------------------------------------------------------------------------------------------------
'''
page_1_layout = html.Div([
    # Row 1 - top bar
    # dbc.Row([topBar]),
    # Row 2 - body
    dbc.Row([
        # Row 2, Col 1 - navigation bar
        dbc.Col([sideBar]),
        # Row 2, col 2 - text description
        dbc.Col([optionGraph]),
        # Row 2, Col 3 - table
        dbc.Col([graphOverview])
    ])
])