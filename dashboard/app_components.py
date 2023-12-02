import dash_bootstrap_components as dbc
import base64
import cvxpy
from dash import html, dcc, dash_table
from models.main import TradeBot


algo = TradeBot()


'''
# ----------------------------------------------------------------------------------------------------------------------
# STYLES
# ----------------------------------------------------------------------------------------------------------------------
'''

top_height = 0
side_bar_width = '12%'
option_width = '20%'


OPTION_ELEMENT = {
    "margin": "1%",
    "font-size": "12px"
}

OPTION_BTN = {
    'margin': '3%',
    'height': '60px',
    'background-color': "#111723",
    'color': 'white',
    'font-size': '12px',
    'verticalAlign': 'middle',
    'border-radius': '15px',
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


MOBILE_PAGE = {
    "position": "fixed",
    "padding": "4px",
    'display': 'flex',
    'flex-direction': 'column',
    'overflow': 'auto',
    "background-color": "#111723",
    'top': 0,
    "left": 0,
    "bottom": 0,
    "width": '100%',
}

# loading sign on the top of the button
LOADING_STYLE = {
    'background': 'white',
    'display': 'flex',
    'justifyContent': 'center',
    'alignItems': 'center',
    'position': "fixed",
    'left': '32%',
    'right': '0%',
    'top': 0,
    'bottom': '0%'
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

spinner_dots = html.Div([
    dcc.Loading(
        id="loading-dots",
        children=[html.Div([html.Div(id="loading-output-dots")])],
        type="circle",
        style=LOADING_STYLE,
        color='black'
    ),
])


spinner_ml = html.Div([
    dcc.Loading(
        id="loading-ml",
        children=[html.Div([html.Div(id="loading-output-ml")])],
        type="circle",
        style=LOADING_STYLE,
        color='black'
    ),
])


spinner_backtest = html.Div([
    dcc.Loading(
        id="loading-backtest",
        children=[html.Div([html.Div(id="loading-output-backtest")])],
        type="circle",
        style=LOADING_STYLE,
        color='black'
    ),
])

# sidebar with navigation
sideBar = html.Div([
    html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
             style={"position": "fixed", 'width': '9%', 'margin-top': '16px'}),
    html.H5("Investment Funnel", style={'color': '#ffd0b3', "position": "fixed", 'top': '7%'}),
    dbc.Nav(
        [
            dbc.NavLink("Market Overview", id='page0', href="/", active="exact"),
            dbc.NavLink("AI Feature Selection", id='page1', href="/page-1", active="exact", n_clicks=0),
            dbc.NavLink("Backtesting", id='page2', href="/page-2", active="exact")
        ],
        vertical=True,
        pills=True,
        style={"position": "fixed", 'top': '9%'}
    )

], style=SIDEBAR_STYLE)


# BACK-TESTING
# ----------------------------------------------------------------------------------------------------------------------
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
        min_date_allowed=algo.min_date,
        max_date_allowed=algo.max_date,
        start_date=algo.min_date,
        style=OPTION_ELEMENT
    ),
    html.P("Testing period for backtest",
           style=SUB_TITLE),
    dcc.DatePickerRange(
        id='picker-test',
        min_date_allowed=algo.min_date,
        max_date_allowed=algo.max_date,
        end_date=algo.max_date,
        style=OPTION_ELEMENT
    ),

    html.P("Portfolio Optimization Model", style=SUB_TITLE),
    dcc.Dropdown(
        id='select-optimization-model',
        options=[
            {'label': value, 'value': value} for value in ['CVaR model', 'Markowitz model']
        ],
        placeholder="Select portfolio optimization model",
        style=OPTION_ELEMENT
    ),

    html.P("Solver", style=SUB_TITLE),
    html.P("MOSEK or ECOS are recommended for CVaR model, for Markowitz model we recommend MOSEK",
           style=DESCRIP_INFO),
    dcc.Dropdown(
        id='select-solver',
        options=[
            {'label': value, 'value': value} for value in cvxpy.installed_solvers()
        ],
        placeholder="Select your installed solver",
        style=OPTION_ELEMENT,
    ),

    html.P("Feature Selection",
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
             style=DESCRIP_INFO),

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
             style=DESCRIP_INFO),
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
    
    html.Div(id='slider-output-container2',
             style=DESCRIP_INFO),
    # part3
    dcc.Slider(250, 2000,
        id='my-slider2',
        step=None,
        marks={
            250: '0.25k',
            500: '0.5k',
            750: '0.75k',
            1000: '1k',
            1250: '1.25k',
            1500: '1.5k',
            1750: '1.75k',
            2000: '2k'},
        value=1000,
    ),

    html.P("Benchmark", style=SUB_TITLE),
    dcc.Dropdown(
        id='select-benchmark',
        options=[
            {'label': value, 'value': value} for value in algo.names
        ],
        placeholder="Select your ETF benchmark",
        multi=True,
        style=OPTION_ELEMENT,
    ),

    dbc.Button('Run Backtest', id='backtestRun', loading_state={'is_loading': 'true'},  style=OPTION_BTN),

 ], style=GRAPH_LEFT)

# Performance
graphResults = html.Div([
    html.Div(id='backtestPerfFig', style=OPTION_ELEMENT),
    html.Div(id='backtestCompFig', style=OPTION_ELEMENT),
    html.Div(id='backtestUniverseFig', style=OPTION_ELEMENT)
], style=GRAPH_RIGHT)


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
        style=OPTION_ELEMENT,
        min_date_allowed=algo.min_date,
        max_date_allowed=algo.max_date,
        start_date=algo.min_date,
        end_date=algo.max_date
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

    dbc.Button('Compute', id='MLRun', style=OPTION_BTN),

], style=GRAPH_LEFT)

selectionBar = html.Div([
    html.H5("Selected assets", style={'text-align': 'left', 'margin-left': '2%'}),
    html.Div(id="AInumber", style={'text-align': 'left', 'margin-left': '2%'}, children="No selected asset."),
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
        style=OPTION_ELEMENT,
        min_date_allowed=algo.min_date,
        max_date_allowed=algo.max_date
    ),

    # Option to search for a fund
    html.P("Find your fund",
           style=SUB_TITLE),
    dcc.Dropdown(
        id='find-fund',
        options=[
            {'label': value, 'value': value} for value in algo.names
        ],
        placeholder="Select here",
        multi=True,
        style=OPTION_ELEMENT,
    ),

    # Button to plot results
    dbc.Button('Show Plot', id='show', style=OPTION_BTN),

], style=GRAPH_LEFT)

# Table
graphOverview = html.Div(id='dotsFig', style=GRAPH_RIGHT)

# Page which shows message for mobile device
mobile_page = html.Div([
    html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
             style={"position": "fixed", 'width': '90%', 'margin-top': '16px', 'right': '5%'}),
    html.H1("Investment Funnel", style={'color': '#ffd0b3', "position": "fixed", 'top': '8%', 'right': '5%'}),
    html.H4("This page is not available on mobile devices. Please use a desktop browser.",
            style={'color': 'white', "position": "fixed", 'top': '20%', 'right': '5%', 'left': '5%'})

], style=MOBILE_PAGE)
