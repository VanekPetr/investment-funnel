import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import pandas as pd
import dash_table
import math
from dash_extensions import Download
import base64
from datetime import date
from dataAnalyser import tickers



'''
# ----------------------------------------------------------------------------------------------------------------------
# STYLES
# ----------------------------------------------------------------------------------------------------------------------
'''

top_height = '7%'
side_bar_width = '10%'

TOPBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "right": 0,
    "height": top_height,
    "background-color": "#111723",
    "textAlign": "right",
}


SIDEBAR_STYLE = {
    "position": "fixed",
    "top": top_height,
    "left": 0,
    "bottom": 0,
    "width": side_bar_width,
    "padding": "2rem 1rem",
    "background-color": "#111723",
}


GRAPH_LEFT = {
    "position": "fixed",
    "left": side_bar_width,
    "top": top_height,
    "width": '20%',
    'bottom': '0%',
    "background-color":  "#d4d5d6",
    "padding": "1rem 1rem",
}

GRAPH_LEFT_TOP = {
    "position": "fixed",
    "left": side_bar_width,
    "top": top_height,
    "width": '20%',
    'bottom': '60%',
    "background-color":  "#d4d5d6",
    "padding": "1rem 1rem",
}

GRAPH_LEFT_SECOND = {
    "position": "fixed",
    "left": side_bar_width,
    "top": "40%",
    "width": '20%',
    'bottom': '45%',
    "background-color":  "#d4d5d6",
    "padding": "1rem 1rem",
}

GRAPH_LEFT_MIDDLE = {
    "position": "fixed",
    "left": side_bar_width,
    "top": '55%',
    "width": '20%',
    'bottom': '25%',
    "background-color":  "#d4d5d6",
    "padding": "1rem 1rem",
}

GRAPH_LEFT_DOWN = {
    "position": "fixed",
    "left": side_bar_width,
    "top": "75%",
    "width": '20%',
    'bottom': '0%',
    #"background-color":  "#8ab4de",
    "padding": "1rem 1rem",
}

GRAPH_RIGHT = {
    "position": "fixed",
    "left": '30%',
    'right': 0,
    "top": top_height,
    'bottom': 0,
    #"background-color":  "#f5e5b5",
    "padding": "1rem 1rem",
}

GRAPH_RIGHT_TOP = {
    "position": "fixed",
    "left": '30%',
    'right': 0,
    "top": top_height,
    'bottom': '47%',
    #"background-color":  "#8ab4de",
    "padding": "1rem 1rem",
}

GRAPH_RIGHT_DOWN = {
    "position": "fixed",
    "left": '30%',
    'right': 0,
    "top": "53%",
    'bottom': '0%',
    #"background-color":  "#f5e5b5",
    "padding": "1rem 1rem",
}


'''
# ----------------------------------------------------------------------------------------------------------------------
# COMPONENTS
# ----------------------------------------------------------------------------------------------------------------------
'''

# GENERAL
# ----------------------------------------------------------------------------------------------------------------------
image_filename = 'assets/ALGO_logo.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

# Top bar with Grundfos logo
topBar = html.Div([
         html.Img(src= 'data:image/png;base64,{}'.format(encoded_image.decode()), style={'height': '85%', 'margin-top': 5, 'margin-right': 10})
            ], style=TOPBAR_STYLE)

# Side bar with navigation
sideBar = html.Div([
        html.P(
            "Navigation", className="lead", style={'color': 'white'}
        ),

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
loading =  dcc.Loading(
            id="loading-1",
            type="default",
            # fullscreen=True,
            parent_style= {'width': '400px','height': '400px', 'margin': '20px'},
            children=html.Div(id="backtestPerfFig")
        )

loading =  dcc.Loading(
            id="loading-1",
            type="default",
            # fullscreen=True,
            parent_style= {'width': '400px','height': '400px','margin':'0', 'top': '50%', 'left': '40%'},
            children=html.Div(id="backtestPerfFig")
        )
        
optionBacktest = html.Div([
    html.H5("BACKTESTING", style={'text-aling': 'left', "position": "fixed", 'top': '10%', 'left': '11%'}),
    html.P("Test your investment strategy with a selected ML model, scenario generation method and the CVaR model for a given training and testing time period",
            style={'text-aling': 'left', "position": "fixed", 'top': '12%', 'left': '11%', 'right':'71%'}),

    html.P("--- Training period ---",
            style={'width': '80%', 'position': 'absolute', 'margin-left': '5%', "top": "40%"}),
    dcc.DatePickerRange(
        id='picker-train',
        style={'position':'absolute', 'top': '46%', 'margin-left': '10%'}
    ),

    html.P("--- Testing period for backtest ---",
            style={'width': '80%', 'position': 'absolute', 'margin-left': '5%', "top": "60%"}),
    dcc.DatePickerRange(
        id='picker-test',
        style={'position':'absolute', 'top': '66%', 'margin-left': '10%'}
    ),

    html.P("--- Feature selection ---",
            style={'width': '80%', 'position': 'absolute', 'margin-left': '5%', "top": "80%"}),
    dcc.Dropdown(
        id='select-ml',
        options=[
            {'label': 'Minimum Spanning Tree', 'value': 'MST'},
            {'label': 'Clustering', 'value': 'Clustering'},
        ],
        placeholder="Select ML method",
        style={'width': '80%', 'position': 'absolute', 'margin-left': '5%', "top": "85%"},
    ),

    html.Div(id='slider-output-container-backtest-ml',
            style={'width': '80%', 'position': 'absolute', 'margin-left': '5%', "top": "95%"}),

], style=GRAPH_LEFT_TOP)

optionBacktest1 = html.Div([
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
        value=2
    ),

    html.Div(id='slider-output-container-backtest',
             style={'width': '80%', 'position': 'absolute', 'margin-left': '5%', "top": "34%"}),

html.P("--- Scenarios ---",
        style={'width': '80%', 'position': 'absolute', 'margin-left': '5%', "top": "65%"}),

dcc.Dropdown(
        id='select-scenarios',
        options=[
            {'label': 'Bootstrapping', 'value': 'Bootstrapping'},
            {'label': 'Monte Carlo', 'value': 'MonteCarlo'}
        ],
        placeholder="Select scenario generation method",
        style={'width': '80%', 'position': 'absolute', 'margin-left': '5%', "top": "80%"},
    ),

], style=GRAPH_LEFT_SECOND)

optionBacktest2 = html.Div([
    dcc.Slider(
        id='my-slider2',
        min=250,
        max=2000,
        step=250,
        value=1000
    ),
    html.Div(id='slider-output-container2',
             style={'width': '80%', 'position': 'absolute', 'margin-left': '5%', "top": "10%"}),

    html.P("--- Benchmark ---",
            style={'width': '80%', 'position': 'absolute', 'margin-left': '5%', "top": "25%"}),
    dcc.Dropdown(
        id='select-benchmark',
        options = [
            {'label': value, 'value': value} for value in tickers
        ],
        placeholder="Select your ETF benchmark",
        multi=True,
        style={'width': '80%', 'position': 'absolute', 'margin-left': '5%', "top": "35%"},
    ),

    html.Button('Run Backtest', id='backtestRun', loading_state={'is_loading' : 'true'},  style={'width': '60%', 'height': 50, 'position':'absolute', 'margin-left': '10%', 'background-color': "#111723", 'color': 'white', "top": "70%"}),
    

], style=GRAPH_LEFT_MIDDLE)



# Table
tableBar = html.Div([
    html.H5("RESULTS:", style={'text-aling': 'left', "position": "fixed", 'top': '77%', 'left': '11%'}),

    html.P("Table for our optimal portfolio",
            style={'width': '80%', 'position': 'absolute', 'margin-left': '5%', "top": "20%"}),
    dash_table.DataTable(id='tableResult',
          columns=[{"name": 'Avg An Ret', "id": 'Avg An Ret'},
                   {"name": 'Std Dev of Ret', "id": 'Std Dev of Ret'},
                   {"name": 'Sharpe R', "id": 'Sharpe R'}],
          #fixed_rows={'headers': True},
          style_table={"position": "fixed",
                       'width':'17%',
                       'margin-left':'1%',
                       'margin-top': '4%',
                       'overflowY':'scroll',
                       'maxHeight':'85%'
          },
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
            style={'width': '80%', 'position': 'absolute', 'margin-left': '5%', "top": "55%"}),
    dash_table.DataTable(id='tableResult-benchmark',
                        columns=[{"name": 'Avg An Ret', "id": 'Avg An Ret'},
                                 {"name": 'Std Dev of Ret', "id": 'Std Dev of Ret'},
                                 {"name": 'Sharpe R', "id": 'Sharpe R'}],
                        # fixed_rows={'headers': True},
                        style_table={"position": "fixed",
                                     'width': '17%',
                                     'margin-left': '1%',
                                     'margin-top': '9%',
                                     'overflowY': 'scroll',
                                     'maxHeight': '85%'
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
], style=GRAPH_LEFT_DOWN)


# Performance
graphPerformance = html.Div([html.Div(loading)], id='backtestPerfFig', style=GRAPH_RIGHT_TOP)

# Composition
graphComposition = html.Div(id='backtestCompFig', style=GRAPH_RIGHT_DOWN)


# AI Feature Selection
# ----------------------------------------------------------------------------------------------------------------------

optionML = html.Div([
    html.H5("Minimum Spanning Tree and Clustering", style={'text-aling': 'left', "position": "fixed", 'top': '10%', 'left': '11%'}),
    html.P("Use one of our developed machine learning algorithms to decrease the number of ETFs in your asset universe. On the right side of this page you can compare outputs of Minimum Spanning Tree and Clustering in created graphs for a selected time period.",
            style={'text-aling': 'left', "position": "fixed", 'top': '13%', 'left': '11%', 'right':'71%'}),

    html.P("--- Time period for feature selection ---",
            style={'width': '80%', 'position': 'absolute', 'margin-left': '5%', "top": "20%"}),
    # Select time period
    dcc.DatePickerRange(
        id='picker-AI',
        style={'position':'absolute', 'top': '23%', 'margin-left': '10%'}
    ),

    # Select MST
    dcc.Dropdown(
        id='mst-dropdown',
        options=[
            {'label': '1 MST run', 'value': 1},
            {'label': '2 MST runs', 'value': 2},
            {'label': '3 MST runs', 'value': 3},
            {'label': '4 MST runs', 'value': 4},
        ],
        placeholder="Select # of MST runs",
        style={'width': '85%', 'position': 'absolute', 'margin-left': '5%', "top": "35%"},
    ),
    # RUN MST
    html.Button('Run MST',
        id='mstRun',
        style={'width': '70%', 'height': 50, 'position':'absolute', 'margin-left': '10%',
               'background-color': "#111723", 'color': 'white', "top": "40%"}),

    # Select clustering
    dcc.Dropdown(
        id='cluster-dropdown',
        options=[
            {'label': '2 Clusters', 'value': 2},
            {'label': '3 Clusters', 'value': 3},
            {'label': '4 Clusters', 'value': 4},
            {'label': '5 Clusters', 'value': 5},
        ],
        placeholder="Select # of clusters",
        style={'width': '85%', 'position': 'absolute', 'margin-left': '5%', "top": "55%"},
    ),
    # RUN Clustering

    html.Button('Run Clustering',
        id='clusterRun',
        style={'width': '70%', 'height': 50, 'position':'absolute', 'margin-left': '10%',
                'background-color': "#111723", 'color': 'white', "top": "60%"}),

], style=GRAPH_LEFT)

# Table
graphML = html.Div(id='mlFig', style=GRAPH_RIGHT_TOP)
graphML2 = html.Div(id='mlFig2', style=GRAPH_RIGHT_DOWN)


# MARKET OVERVIEW
# ----------------------------------------------------------------------------------------------------------------------

optionGraph = html.Div([
    html.H5("INVESTMENT FUNNEL", style={'text-aling': 'left', "position": "fixed", 'top': '10%', 'left': '11%'}),
    # Introduction
    html.P("This tool is used for students and asset managers to get better overview of the ETF market, to experiment with a different investment techniques & algorithms and finally to backtest their investment strategies.",
            style={'text-aling': 'left', "position": "fixed", 'top': '13%', 'left': '11%', 'right':'71%'}),
    # Description of the first page
    html.P("On this first page you can select different time periods and compare ETF market in terms of gains and risks. Risk is here represented by annualised standard deviation of returns and gain in terms of annualised average returns. The largest possible time interval is already pre-selected for you. Have you already found your favourite ETF?",
            style={'text-aling': 'left', "position": "fixed", 'top': '20%', 'left': '11%', 'right':'71%'}),
    # Description of the first page
    html.P("Do you have your own investment strategy but before you apply it you want to cleverly decrease the number of all possible ETF choices? The second page of this tool will help you with a use of machine learning algorithms as Clustering or Minimum Spanning Tree.",
            style={'text-aling': 'left', "position": "fixed", 'top': '30%', 'left': '11%', 'right':'71%'}),
    # Description of the last page
    html.P("In case you want to use this tool to create the best trading strategy, go to the last page named Backtesting. There you can configurate your optimal strategy with a usage of the stochastic CVaR model which selects the optimal portfolio maximizing expected returns given some risk benchmark. Select by yourself your prefered time periods for training of the model and for backtesting as well as your favourite models for feature selection and scenario generation. Lastly, do not forget about your risk benchmark which can be for example SPY ETF. Our optimal CVaR model will rebalance every 4 weeks so you can be sure that you are always on the right track for the best results.",
            style={'text-aling': 'left', "position": "fixed", 'top': '39%', 'left': '11%', 'right':'71%'}),

    html.P("--- Selected dates for market overview ---",
            style={'width': '80%', 'position': 'absolute', 'margin-left': '5%', "top": "72%"}),
    # Date picker for plotting
    dcc.DatePickerRange(
        id='picker-show',
        style={'position':'absolute', 'top': '75%', 'margin-left': '10%'}
    ),
    # Button to plot results
    html.Button('Show Plot',
                id='show',
                style={'width': '70%', 'height': 50, 'position':'absolute', 'margin-left': '10%',
                        'background-color': "#111723", 'color': 'white', "top": "85%"}),
], style=GRAPH_LEFT)


# Table
graphOverview = html.Div(id='dotsFig', style=GRAPH_RIGHT)
