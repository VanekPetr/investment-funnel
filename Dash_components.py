import dash_bootstrap_components as dbc
import base64
from dash import html
from dash import dcc
from dash import dash_table
from models.main import names


'''
# ----------------------------------------------------------------------------------------------------------------------
# STYLES
# ----------------------------------------------------------------------------------------------------------------------
'''

top_height = 0
side_bar_width = '12%'
option_width = '20%'

TOPBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "right": 0,
    "height": top_height,
    "background-color": "#111723",
    "textAlign": "right",
}

OPTION_ELEMENT = {
    "margin": "1%",
    "font-size": "12px"
}

OPTION_BTN = {
    'margin': '2%',
    'height': '32px',
    'background-color': "#111723", 
    'color': 'white',
    'font-size': '12px'
}

MAIN_TITLE = {
    "text-align": "left",
    "margin": "2%",
    "margin-top": '16px',
    "font-size": "16px",
    "font-weight": "600",
}

SUB_TITLE = {
    "text-align": "left",
    "margin-top": "6%",
    "margin-bottom": "1%",
    "margin-left": "2%",
    "font-size": "12px",
    "font-weight": "500",
    "color": "#191919"
}

DESCRIP_INFO = {
    "text-aling": "left",
    "margin": "2%",
    "font-size": "12px",
    "color": "#5d5d5d"
}

SIDEBAR_STYLE = {
    "position": "fixed",
    'top': 0,
    "left": 0,
    "bottom": 0,
    "width": side_bar_width,
    "padding": "1rem",
    "background-color": "#111723",
    # "li:hover" : "#EF9761"
    'display': 'flex', 
    'flex-direction': 'column',
    'overflow': 'auto',
}

NAV_BTN = {
    'a:color': 'white',

}

GRAPH_LEFT = {
    "position": "fixed",
    "left": side_bar_width,
    "top": 0,
    "width": option_width,
    'bottom': '0%',
    "background-color":  "#d4d5d6",
    "padding": "8px",
    'display': 'flex', 
    'flex-direction': 'column',
    # 'flex-flow': 'column wrap'
    'overflow': 'auto',
}

GRAPH_RIGHT = {
    "position": "fixed",
    "left": '32%',
    'right': '0%',
    "top": 0,
    'bottom': '0%',
    "padding": "4px",
    'display': 'flex', 
    'flex-direction': 'column',
    'overflow': 'auto',
}


'''
# ----------------------------------------------------------------------------------------------------------------------
# COMPONENTS
# ----------------------------------------------------------------------------------------------------------------------
'''

# GENERAL
# ----------------------------------------------------------------------------------------------------------------------
image_filename = 'assets/ALGO_logo.png'  # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

# Top bar with AlgoStrata logo
topBar = html.Div([
         html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
                  style={'height': '16px', 'margin': '12px', 'margin-right': '16px'})
            ], style=TOPBAR_STYLE)

# sidebar with navigation
sideBar = html.Div([
        html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
                 style={'width': '144px', 'height': '12px', 'margin-top': '16px', 'margin-bottom': '36px'}),
        dbc.Nav(
            [
                dbc.NavLink("Market Overview", id='page0', href="/", active="exact"),
                dbc.NavLink("AI Feature Selection", id='page1', href="/page-1", active="exact", n_clicks=0),
                dbc.NavLink("Backtesting", id='page2', href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ], style=SIDEBAR_STYLE,
)


# BACK-TESTING
# ----------------------------------------------------------------------------------------------------------------------
# Loading 
loading = dcc.Loading(
    id="loading-1",
    type="default",
    children=html.Div(id="backtestPerfFig")
)

optionBacktest = html.Div([
    # part1
    html.H5("Backtesting", style=MAIN_TITLE),
    html.P("Test your investment strategy with a selected ML model, scenario generation method and the CVaR model " +
           "for a given training and testing time period",
           style=DESCRIP_INFO),
    html.P("Training period",
           style=SUB_TITLE),
    dcc.DatePickerRange(
        id='picker-train',
        style=OPTION_ELEMENT
    ),
    html.P("Testing period for backtest",
           style=SUB_TITLE),
    dcc.DatePickerRange(
        id='picker-test',
        style=OPTION_ELEMENT
    ),
    html.P("Feature selection",
           style=SUB_TITLE),
    dcc.Dropdown(
        id='select-ml',
        options=[
            {'label': 'Minimum Spanning Tree', 'value': 'MST'},
            {'label': 'Clustering', 'value': 'Clustering'},
        ],
        placeholder="Select ML method",
        style=OPTION_ELEMENT,
    ),
    html.Div(id='slider-output-container-backtest-ml',
             style=OPTION_ELEMENT),

    # part2
    dcc.Slider(
        id='slider-backtest-ml',
        min=1,
        max=5,
        step=1,
        value=2
    ),
    dcc.Slider(
        id='slider-backtest',
        min=1,
        max=20,
        step=1,
        value=2,
    ),
    html.Div(id='slider-output-container-backtest',
             style=OPTION_ELEMENT),
    html.P("Scenarios",
           style=SUB_TITLE),
    dcc.Dropdown(
            id='select-scenarios',
            options=[
                {'label': 'Bootstrapping', 'value': 'Bootstrapping'},
                {'label': 'Monte Carlo', 'value': 'MonteCarlo'}
            ],
            placeholder="Select scenario generation method",
            style=OPTION_ELEMENT,
        ),
    
    # part3
    dcc.Slider(
        id='my-slider2',
        min=250,
        max=2000,
        step=250,
        value=1000
    ),
    html.Div(id='slider-output-container2',
             style=OPTION_ELEMENT),
    html.P("Benchmark",
           style=SUB_TITLE),
    dcc.Dropdown(
        id='select-benchmark',
        options=[
            {'label': value, 'value': value} for value in names
        ],
        placeholder="Select your ETF benchmark",
        multi=True,
        style=OPTION_ELEMENT,
    ),
    html.Button('Run Backtest', id='backtestRun', loading_state={'is_loading': 'true'},  style=OPTION_BTN),

 ], style=GRAPH_LEFT)

# Table
tableBar = html.Div([
    html.H5("Results", style={'text-aling': 'left', 'margin-left': '2%'}),

    html.P("Table for our optimal portfolio",
           style={'width': '80%',  'margin-left': '2%', }),
    dash_table.DataTable(id='tableResult',
                         columns=[{"name": 'Avg An Ret', "id": 'Avg An Ret'},
                                  {"name": 'Std Dev of Ret', "id": 'Std Dev of Ret'},
                                  {"name": 'Sharpe R', "id": 'Sharpe R'}],
                         # fixed_rows={'headers': True},
                         style_table={'width': '48%', 'margin': '2%'},
                         style_cell={'textAlign': 'center'},
                         style_as_list_view=True,
                         style_header={'fontWeight': 'bold'},
                         style_cell_conditional=[
                             {
                                 'if': {'column_id': c},
                                 'textAlign': 'left'
                             } for c in ['variable', 'Group name', 'subgroup name', 'Attribute text']

                         ]),

    html.P("Table for Benchmark",
           style={'width': '80%', 'margin-left': '2%'}),
    dash_table.DataTable(id='tableResult-benchmark',
                         columns=[{"name": 'Avg An Ret', "id": 'Avg An Ret'},
                                  {"name": 'Std Dev of Ret', "id": 'Std Dev of Ret'},
                                  {"name": 'Sharpe R', "id": 'Sharpe R'}],
                         # fixed_rows={'headers': True},
                         style_table={
                                    'width': '48%',
                                    'margin': '2%',
                                    #  'overflowY': 'scroll',
                                    #  'maxHeight': '85%'
                                     },
                         style_cell={'textAlign': 'center'},
                         style_as_list_view=True,
                         style_header={'fontWeight': 'bold'},
                         style_cell_conditional=[
                            {
                                'if': {'column_id': c},
                                'textAlign': 'left'
                            } for c in ['variable', 'Group name', 'subgroup name', 'Attribute text']
                         ])
], style=OPTION_ELEMENT)

# Performance
graphResults = html.Div([
    html.Div([html.Div(loading)], id='backtestPerfFig', style=OPTION_ELEMENT),
    html.Div(id='backtestCompFig', style=OPTION_ELEMENT),
    tableBar
], style=GRAPH_RIGHT)

# graphPerformance = html.Div([html.Div(loading)], id='backtestPerfFig', style=GRAPH_RIGHT_TOP)

# Composition
# graphComposition = html.Div(id='backtestCompFig', style=GRAPH_RIGHT_DOWN)


# AI Feature Selection
# ----------------------------------------------------------------------------------------------------------------------
optionML = html.Div([
    html.H5("Minimum Spanning Tree & Clustering", style=MAIN_TITLE),
    html.P("Use machine learning algorithms to decrease the number of ETFs in your asset universe.",
           style=DESCRIP_INFO),

    html.P("Time period for feature selection", style=SUB_TITLE),
    # Select time period
    dcc.DatePickerRange(
        id='picker-AI',
        style=OPTION_ELEMENT
    ),

    html.P("AI/ML model", style=SUB_TITLE),
    # Select MST
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'Minimum Spanning Tree', 'value': 'MST'},
            {'label': 'Clustering', 'value': 'Cluster'}
        ],
        placeholder="Select algorithm",
        style=OPTION_ELEMENT,
    ),

    html.P("# of Clusters or # of MST runs", style=SUB_TITLE),
    # Select clustering
    dcc.Dropdown(
        id='ML-num-dropdown',
        options=[
            {'label': '2', 'value': 2},
            {'label': '3', 'value': 3},
            {'label': '4', 'value': 4},
            {'label': '5', 'value': 5},
        ],
        placeholder="Select number",
        style=OPTION_ELEMENT,
    ),
    # RUN Clustering

    html.Button('Compute',
                id='MLRun',
                style=OPTION_BTN),
], style=GRAPH_LEFT)

selectionBar = html.Div([
    html.H5("Selected assets", style={'text-align': 'left', 'margin-left': '2%'}),
    html.Div(id="AInumber", style={'text-align': 'left', 'margin-left': '2%'}),
    dash_table.DataTable(id='AIResult',
                         columns=[{"name": 'Name', "id": 'Name'},
                                  {"name": 'ISIN', "id": 'ISIN'},
                                  {"name": 'Sharpe Ratio', "id": 'Sharpe Ratio'},
                                  {"name": 'Annual Returns', "id": 'Average Annual Returns'},
                                  {"name": 'STD', "id": 'Standard Deviation of Returns'}],
                         style_table={'width': '48%', 'margin': '2%'},
                         style_cell={'textAlign': 'center'},
                         style_as_list_view=True,
                         style_header={'fontWeight': 'bold'},
                         style_cell_conditional=[
                             {
                                 'if': {'column_id': c},
                                 'textAlign': 'left'
                             } for c in ['variable', 'Group name', 'subgroup name', 'Attribute text']

                         ])
], style=OPTION_ELEMENT)

# AI Feature selection graph
graphML = html.Div([
    html.Div(id='mlFig', style=OPTION_ELEMENT),
    selectionBar
], style=GRAPH_RIGHT)


# MARKET OVERVIEW
# ----------------------------------------------------------------------------------------------------------------------
optionGraph = html.Div([
    html.H5("Investment Funnel", style=MAIN_TITLE),
    html.P("Selected dates for market overview",
           style=SUB_TITLE),
    # Date picker for plotting
    dcc.DatePickerRange(
        id='picker-show',
        style=OPTION_ELEMENT
    ),

    # Option to search for a fund
    html.P("Find your fund",
           style=SUB_TITLE),
    dcc.Dropdown(
        id='find-fund',
        options=[
            {'label': value, 'value': value} for value in names
        ],
        placeholder="Select here",
        multi=True,
        style=OPTION_ELEMENT,
    ),

    # Button to plot results
    html.Button('Show Plot',
                id='show',
                style=OPTION_BTN),


], style=GRAPH_LEFT)

# Table
graphOverview = html.Div(id='dotsFig', style=GRAPH_RIGHT)
