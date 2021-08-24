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
    "margin" : "1%",
    "font-size" : "12px"
}

OPTION_BTN = {
    'margin': '2%',
    'height': '32px',
    'background-color': "#111723", 
    'color': 'white',
    'font-size' : '12px'
}

MAIN_TITLE = {
    "text-aling": "left",
    "margin" : "2%",
    "margin-top":'16px',
    "font-size" : "16px", 
    "font-weight" : "600",
}

SUB_TITLE = {
    "text-aling": "left",
    "margin-top" : "3%",
    "margin-bottom" : "1%",
    "margin-left" : "2%",
    "font-size" : "12px", 
    "font-weight" : "500",
    "color" : "#191919"
}

DESCRIP_INFO = {
    "text-aling": "left",
    "margin" : "2%",
    "font-size" : "12px", 
    "color" : "#5d5d5d"
}

SIDEBAR_STYLE = {
    "position": "fixed",
    'top':0,
    "left": 0,
    "bottom": 0,
    "width": side_bar_width,
    "padding": "1rem",
    "background-color": "#111723",
    # "li:hover" : "#EF9761"
    'display': 'flex', 
    'flex-direction':'column', 
    'overflow' : 'auto',
}

NAV_BTN = {
    'a:color' : 'white',

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
    'flex-direction':'column', 
    # 'flex-flow': 'column wrap'
    'overflow' : 'auto',
}

GRAPH_RIGHT = {
    "position": "fixed",
    "left": '32%',
    'right': '0%',
    "top": 0,
    'bottom': '0%',
    "padding": "4px",
    'display': 'flex', 
    'flex-direction':'column', 
    'overflow' : 'auto',
}

SUB_GRAPH_RIGHT =  {
    "margin" : "1%",
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
         html.Img(src= 'data:image/png;base64,{}'.format(encoded_image.decode()), style={'height': '16px', 'margin':'12px', 'margin-right': '16px'})
            ], style=TOPBAR_STYLE)

# Side bar with navigation
sideBar = html.Div([
        html.Img(src= 'data:image/png;base64,{}'.format(encoded_image.decode()), style={'width': '144px', 'height': '12px', 'margin-top' : '16px', 'margin-bottom':'36px'}),
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
            children=html.Div(id="backtestPerfFig")
        )

optionBacktest = html.Div([
    #part1
    html.H5("Backtesting", style=MAIN_TITLE),
    html.P("Test your investment strategy with a selected ML model, scenario generation method and the CVaR model for a given training and testing time period",
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

    #part2
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
    
    #part3
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
        options = [
            {'label': value, 'value': value} for value in tickers
        ],
        placeholder="Select your ETF benchmark",
        multi=True,
        style=OPTION_ELEMENT,
    ),
    html.Button('Run Backtest', id='backtestRun', loading_state={'is_loading' : 'true'},  style=OPTION_BTN),

 ], style=GRAPH_LEFT)



# Table
tableBar = html.Div([
    html.H5("Results", style={'text-aling': 'left', 'margin-left': '2%'}),

    html.P("Table for our optimal portfolio",
            style={'width': '80%',  'margin-left': '2%',}),
    dash_table.DataTable(id='tableResult',
          columns=[{"name": 'Avg An Ret', "id": 'Avg An Ret'},
                   {"name": 'Std Dev of Ret', "id": 'Std Dev of Ret'},
                   {"name": 'Sharpe R', "id": 'Sharpe R'}],
          #fixed_rows={'headers': True},
          style_table={
                       'width':'48%',
                       'margin':'2%',
                    #    'overflowY':'scroll',
                    #    'maxHeight':'85%'
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
            style={'width': '80%','margin-left': '2%'}),
    dash_table.DataTable(id='tableResult-benchmark',
                        columns=[{"name": 'Avg An Ret', "id": 'Avg An Ret'},
                                 {"name": 'Std Dev of Ret', "id": 'Std Dev of Ret'},
                                 {"name": 'Sharpe R', "id": 'Sharpe R'}],
                        # fixed_rows={'headers': True},
                        style_table={
                                    'width':'48%',
                                    'margin':'2%',
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

    html.P("Time period for feature selection",
            style=SUB_TITLE),
    # Select time period
    dcc.DatePickerRange(
        id='picker-AI',
        style=OPTION_ELEMENT
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
        style=OPTION_ELEMENT,
    ),
    # RUN MST
    html.Button('Run MST',
        id='mstRun',
        style=OPTION_BTN),

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
        style=OPTION_ELEMENT,
    ),
    # RUN Clustering

    html.Button('Run Clustering',
        id='clusterRun',
        style=OPTION_BTN),

], style=GRAPH_LEFT)

# Table
graphML = html.Div([
    html.Div(id='mlFig', style=SUB_GRAPH_RIGHT), #MST Graph
    html.Div(id='mlFig2', style=SUB_GRAPH_RIGHT), #Clust Graph
], style=GRAPH_RIGHT)


# MARKET OVERVIEW
# ----------------------------------------------------------------------------------------------------------------------

optionGraph = html.Div([
    html.H5("Investment Funnel", style=MAIN_TITLE),
    # # Introduction
    # html.P("This tool is used for students and asset managers to get better overview of the ETF market, to experiment with a different investment techniques & algorithms and finally to backtest their investment strategies.",
    #         style={'text-aling': 'left', "position": "fixed", 'top': '13%', 'left': '11%', 'right':'71%'}),
    # # Description of the first page
    # html.P("On this first page you can select different time periods and compare ETF market in terms of gains and risks. Risk is here represented by annualised standard deviation of returns and gain in terms of annualised average returns. The largest possible time interval is already pre-selected for you. Have you already found your favourite ETF?",
    #         style={'text-aling': 'left', "position": "fixed", 'top': '20%', 'left': '11%', 'right':'71%'}),
    # # Description of the first page
    # html.P("Do you have your own investment strategy but before you apply it you want to cleverly decrease the number of all possible ETF choices? The second page of this tool will help you with a use of machine learning algorithms as Clustering or Minimum Spanning Tree.",
    #         style={'text-aling': 'left', "position": "fixed", 'top': '30%', 'left': '11%', 'right':'71%'}),
    # # Description of the last page
    # html.P("In case you want to use this tool to create the best trading strategy, go to the last page named Backtesting. There you can configurate your optimal strategy with a usage of the stochastic CVaR model which selects the optimal portfolio maximizing expected returns given some risk benchmark. Select by yourself your prefered time periods for training of the model and for backtesting as well as your favourite models for feature selection and scenario generation. Lastly, do not forget about your risk benchmark which can be for example SPY ETF. Our optimal CVaR model will rebalance every 4 weeks so you can be sure that you are always on the right track for the best results.",
    #         style={'text-aling': 'left', "position": "fixed", 'top': '39%', 'left': '11%', 'right':'71%'}),

    html.P("Selected dates for market overview",
            style=SUB_TITLE),
    # Date picker for plotting
    dcc.DatePickerRange(
        id='picker-show',
        style=OPTION_ELEMENT
    ),
    # Button to plot results
    html.Button('Show Plot',
                id='show',
                style=OPTION_BTN),
], style=GRAPH_LEFT)


# Table
graphOverview = html.Div(id='dotsFig', style=GRAPH_RIGHT)
