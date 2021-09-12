import os
import dash
from dash_html_components import Br
from dash_html_components.Font import Font
import dash_table

import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import dash_gif_component as gif
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html

from copy import deepcopy
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output
from pandas.core.common import SettingWithCopyWarning

import warnings
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

###################################################

# Load data
list_stocks = ['AAPL', 'MSFT', 'TSLA', 'AMZN']
df_price = pd.read_csv(os.getcwd() + '\\assets\\price.csv')
df_price = df_price.set_index('Date')
df_shareholders = pd.read_csv(os.getcwd() + '\\assets\\shareholders.csv')
df_mc = pd.read_csv(os.getcwd() + '\\assets\\monte-carlo.csv')
df_esg = pd.read_csv(os.getcwd() + '\\assets\\esg.csv')
cov_matrix = pd.read_csv(os.getcwd() + '\\assets\\cov_portfolio.csv')
corr_matrix = pd.read_csv(os.getcwd() + '\\assets\\corr_portfolio.csv')
portfolios = pd.read_csv(os.getcwd() + '\\assets\\portfolios.csv')
news = pd.read_csv(os.getcwd() + '\\assets\\news.csv')
copulas_table = pd.read_csv(os.getcwd() + '\\assets\\copulas_table.csv')
copulas_plots = pd.read_csv(os.getcwd() + '\\assets\\copulas_plots.csv')
df_additional = pd.read_csv(os.getcwd() + '\\assets\\additional.csv')
image_location = os.getcwd() + '\\assets\\logo.png'

# Initialize the app
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True


def get_options(list_stocks):
    dict_list = []
    for i in list_stocks:
        dict_list.append({'label': i, 'value': i})
    return dict_list

###################################################

app.layout = html.Div(
    children=[
        html.Div(className='row',
                 children=[
                    html.Div(className='four columns div-user-controls',
                             children=[
                                 html.Div(html.Img(src=app.get_asset_url('logo.png')), style={'text-align':'center'}),
                                 html.Hr(),
                                 html.H2('Financial Dashboard for Stock Analysis'),
                                 html.P('Pick one stock from the menu below.'),
                                 html.Div(
                                     className='div-for-dropdown',
                                     children=[
                                         dcc.Dropdown(id='stockselector', options=get_options(list_stocks),
                                                      multi=False, value=list_stocks[0], clearable=False,
                                                      className='stockselector'
                                                      ),
                                     ],
                                     style={'color': '#1E1E1E'}),
                                 html.H6('Valuation Method : Monte-Carlo'),
                                 html.Div(id='expected-price', style={'font-size':17, 'text-align':'left', 'font-style':'italic'}),
                                 html.Br(),
                                 html.H6('Major Shareholders'),
                                 html.Div(id='major-shareholders'),
                                 html.Br(),
                                 html.H6('ESG Analysis'),
                                 html.Div(id='esg-analysis'),
                                 html.Br(),
                                 html.H6('Weekly News'),
                                 html.Div(id='news'),
                                 html.Br(),
                                 html.H6('Top 5 Portfolios'),
                                 html.Div(id='portfolios-matrix'),
                                 html.Br(),
                                 html.H6('Portfolio Covariance Matrix'),
                                 html.Div(id='cov-matrix'),
                                 html.Br(),
                                 html.H6('Portfolio Correlation Matrix'),
                                 html.Div(id='corr-matrix'),
                                 html.Br(),
                                 html.H6('Copulas Metrics'),
                                 html.Div(id='copulas-table-frank'),
                                 html.Br(),
                                 html.Div(id='copulas-table-clayton'),
                                 html.Br(),
                                 html.Div(id='copulas-table-gumbel'),
                                 html.Br(),
                                 html.Div([
                                     gif.GifPlayer(
                                         gif=app.get_asset_url('trading.gif'),
                                         still=app.get_asset_url('trading-1.png')
                                     )
                                 ], style={'text-align':'center'}),
                                ]
                             ),
                    html.Div(className='eight columns div-for-charts bg-grey',
                             children=[
                                 html.Div(id='stock-title', style={'font-size':30, 'text-align':'right', 'font-style':'bold', 'background-color':'#111', 'color':'white'}),
                                 dcc.Graph(id='historic-prices', config={'displayModeBar': False}, animate=True),
                                 dcc.Graph(id='historic-returns', config={'displayModeBar': False}, animate=True),
                                 dcc.Graph(id='qq-plots', config={'displayModeBar': False}, animate=True),
                                 dcc.Graph(id='mc-frequences', config={'displayModeBar': False}),
                                 dcc.Graph(id='mc-trials', config={'displayModeBar': False}),
                                 dcc.Graph(id='copulas-plots', config={'displayModeBar': False}, animate=True),
                                 dcc.Graph(id='portfolio-scatter', config={'displayModeBar': False}),
                             ])
                              ])
        ]

)

###################################################

# Callback for historic prices
@app.callback(Output('historic-prices', 'figure'),
              [Input('stockselector', 'value')])
def update_graph(selected_dropdown_value):
    trace1 = []
    df_sub = deepcopy(df_price)
    trace1.append(go.Scatter(x=df_sub[df_sub['Ticker'] == selected_dropdown_value].index,
                             y=df_sub[df_sub['Ticker'] == selected_dropdown_value]['Close'],
                             mode='lines',
                             opacity=0.7,
                             name=selected_dropdown_value,
                             textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(
                  colorway=["#5E0DAC"],
                  template='plotly_dark',
                  hovermode='x',
                  autosize=True,
                  title={'text': '<i>Stock Prices</i>', 'font': {'color': 'white', 'size':30}, 'x': 0.5},
                  xaxis={'range': [df_sub.index.min(), df_sub.index.max()]},
                  yaxis={'range': [df_sub[df_sub['Ticker'] == selected_dropdown_value]['Close'].min(), df_sub[df_sub['Ticker'] == selected_dropdown_value]['Close'].max()]},
                  yaxis_title={'text': 'Price ($)', 'font': {'color': 'white'}},
              ),

              }
    return figure

# Callback for daily historic returns (arithmetic)
@app.callback(Output('historic-returns', 'figure'),
              [Input('stockselector', 'value')])
def update_change(selected_dropdown_value):
    trace2 = []
    df_sub = deepcopy(df_price)
    df_sub = df_sub[df_sub['Ticker'] == selected_dropdown_value]
    df_sub = df_sub[2:]
    trace2.append(go.Scatter(x=df_sub.index,
                             y=df_sub['Return'],
                             mode='lines',
                             opacity=0.7,
                             name=selected_dropdown_value,
                             textposition='bottom center'))
    traces = [trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(
                  colorway=['#FFF400'],
                  template='plotly_dark',
                  hovermode='x',
                  autosize=True,
                  title={'text': '<i>Daily Returns</i>', 'font': {'color': 'white', 'size':30}, 'x': 0.5},
                  xaxis={'range': [df_sub.index.min(), df_sub.index.max()]},
                  yaxis={'range': [df_sub[df_sub['Ticker'] == selected_dropdown_value]['Return'].min(), df_sub[df_sub['Ticker'] == selected_dropdown_value]['Return'].max()]},
                  yaxis_title={'text': 'Return (%)', 'font': {'color': 'white'}},
              ),
            }
    return figure


# Callback for Monte-Carlo frequences
@app.callback(Output('mc-frequences', 'figure'),
              [Input('stockselector', 'value')])
def update_mc_histogram(selected_dropdown_value):
    trace3 = []
    sub_df = df_mc[df_mc['Ticker'] == selected_dropdown_value]
    closing_prices = eval(sub_df['closing_prices'].iloc[0])
    trace3.append(go.Histogram(x=closing_prices,
                             opacity=0.7,
                             marker=dict(color="#4CB391")))
    traces = [trace3]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(
                  barmode='overlay',
                  template='plotly_dark',
                  autosize=True,
                  title={'text': '<i>Monte-Carlo Frequences</i>', 'font': {'color': 'white', 'size':30}, 'x': 0.5},
                  yaxis_title={'text': 'Number of Trials', 'font': {'color': 'white'}},
                  xaxis_title={'text': 'Price ($)', 'font': {'color': 'white'}},
              ),
            }
    return figure

# Callback for Monte-Carlo trials
@app.callback(Output('mc-trials', 'figure'),
              [Input('stockselector', 'value')])
def update_mc_scatter(selected_dropdown_value):
    trace4 = []
    sub_df = df_mc[df_mc['Ticker'] == selected_dropdown_value]
    price_trials = eval(sub_df['price_trials'].iloc[0])

    for i in range(500):
        trace4.append(go.Scatter(x=np.array(range(252)),
                                y=price_trials[i],
                                mode='lines',
                                opacity=0.7,
                                name=selected_dropdown_value,
                                textposition='bottom center'))
    traces = [trace4]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(
                  template='plotly_dark',
                  hovermode='x',
                  autosize=True,
                  title={'text': '<i>Monte-Carlo Trials</i>', 'font': {'color': 'white', 'size':30}, 'x': 0.5},
                  showlegend=False,
                  yaxis={'autorange':True},
                  yaxis_title={'text': 'Return ($)', 'font': {'color': 'white'}},
                  xaxis_title={'text': 'Trading Days', 'font': {'color': 'white'}},
              ),
            }
    return figure

# Callback for portfolio scatter
@app.callback(Output('portfolio-scatter', 'figure'),
             [Input('stockselector', 'value')])
def update_portfolio_scatter(selected_dropdown_value):
    trace5 = []
    trace5.append(go.Scatter(x=portfolios['Volatility'],
                             y=portfolios['Returns'],
                             opacity=0.7,
                             mode='markers',
                             textposition='bottom center'))
    traces = [trace5]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(
                  colorway=['#375CB1'],
                  template='plotly_dark',
                  hovermode='x',
                  autosize=True,
                  title={'text': '<i>Portfolio Analysis</i>', 'font': {'color': 'white', 'size':30}, 'x': 0.5},
                  yaxis_title={'text': 'Returns', 'font': {'color': 'white'}},
                  xaxis_title={'text': 'Volatility', 'font': {'color': 'white'}},
              ),
            }
    return figure

# Callback for copulas scatter subplots
@app.callback(Output('copulas-plots', 'figure'),
             [Input('stockselector', 'value')])
def update_copulas(selected_dropdown_value):
    tmp_list = deepcopy(list_stocks)
    tmp_list.remove(selected_dropdown_value)
    list_titles = ["Frank", "Clayton", "Gumbel"]
    sub_df = copulas_plots[copulas_plots['Ticker'] == selected_dropdown_value]
    fig = make_subplots(rows=3, cols=3, subplot_titles=[list_titles[0], list_titles[1], list_titles[2]])
    for i in range(len(tmp_list)):
        sub_df_stock = sub_df[sub_df['Versus'] == tmp_list[i]]
        uf = eval(sub_df_stock['uf'].iloc[0])
        vf = eval(sub_df_stock['vf'].iloc[0])
        uc = eval(sub_df_stock['uc'].iloc[0])
        vc = eval(sub_df_stock['vc'].iloc[0])
        ug = eval(sub_df_stock['ug'].iloc[0])
        vg = eval(sub_df_stock['vg'].iloc[0])
        fig.add_trace(go.Scatter(x=uf,
                                y=vf,
                                opacity=0.7,
                                mode='markers',
                                textposition='bottom center',
                                marker={'size':2}),
                                row=i+1, col=1)
        fig.add_trace(go.Scatter(x=uc,
                                y=vc,
                                opacity=0.7,
                                mode='markers',
                                textposition='bottom center',
                                marker={'size':2}),
                                row=i+1, col=2)
        fig.add_trace(go.Scatter(x=ug,
                                y=vg,
                                opacity=0.7,
                                mode='markers',
                                textposition='bottom center',
                                marker={'size':2}),
                                row=i+1, col=3)
    fig.update_layout(
        template='plotly_dark',
        hovermode='x',
        title={'text': '<i>Copulas Analysis</i>', 'font': {'color': 'white', 'size':30}, 'x': 0.5},
        showlegend=False
    )
    fig.update_yaxes(title_text=f"{tmp_list[0]}", row=1, col=1, title_font={'size':12})
    fig.update_yaxes(title_text=f"{tmp_list[1]}", row=2, col=1, title_font={'size':12})
    fig.update_yaxes(title_text=f"{tmp_list[2]}", row=3, col=1, title_font={'size':12})
    return fig

# Callback for QQ Probability Plots
@app.callback(Output('qq-plots', 'figure'),
             [Input('stockselector', 'value')])
def update_qq_plots(selected_dropdown_value):
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Normal Probabilites', 'Student Probabilities'])
    sub_df = df_additional[df_additional['Ticker'] == selected_dropdown_value]
    x_value_normal = eval(sub_df['normal_x'].iloc[0])
    y_value_normal = eval(sub_df['normal_y'].iloc[0])
    x_value_student = eval(sub_df['student_x'].iloc[0])
    y_value_student = eval(sub_df['student_y'].iloc[0])
    fig.add_trace(go.Scatter(x=x_value_normal,
                             y=y_value_normal,
                             opacity=0.7,
                             mode='markers',
                             textposition='bottom center'),
                             row=1, col=1)
    fig.add_trace(go.Scatter(x=x_value_student,
                             y=y_value_student,
                             opacity=0.7,
                             mode='markers',
                             textposition='bottom center'),
                             row=1, col=2)
    fig.update_layout(
        template='plotly_dark',
        hovermode='x',
        autosize=True,
        title={'text': '<i>QQ-plots (daily returns)</i>', 'font': {'color': 'white', 'size':30}, 'x': 0.5},
        showlegend=False
    )
    fig.update_xaxes(title_text="Theoritical Quantities", row=1, col=1)  
    fig.update_xaxes(title_text="Theoritical Quantities", row=1, col=2)
    fig.update_yaxes(title_text="Ordered Values", row=1, col=1)  
    fig.update_yaxes(title_text="Ordered Values", row=1, col=2)  
    return fig

###################################################

# Callback for Stock's major shareholders
@app.callback(Output('major-shareholders', 'children'),
              [Input('stockselector', 'value')])
def update_shareholders(selected_dropdown_value):
    sub_df = df_shareholders[df_shareholders['Ticker'] == selected_dropdown_value]
    sub_df = sub_df.drop(columns=['Ticker'])
    sub_df = sub_df.round(4)
    table = dash_table.DataTable(
        id='shareholders_table',
        columns=[{"name": i, "id": i} 
                 for i in sub_df.columns],
        data=sub_df.to_dict('records'),
        style_cell={'textAlign':'left', 'font-family':'Open Sans Light'},
        style_header={'backgroundColor':'darkblue', 'font-weight':'bold'},
        style_data=dict(backgroundColor="dimgrey")
    )
    return table

# Callback for ESG Analysis
@app.callback(Output('esg-analysis', 'children'),
              [Input('stockselector', 'value')])
def update_esg(selected_dropdown_value):
    sub_df = df_esg[df_esg['Ticker'] == selected_dropdown_value]
    sub_df = sub_df.drop(columns=['Ticker'])
    table = dash_table.DataTable(
        id='esg_table',
        columns=[{"name": i, "id": i} 
                 for i in sub_df.columns],
        data=sub_df.to_dict('records'),
        style_cell={'textAlign':'left', 'font-family':'Open Sans Light'},
        style_header={'backgroundColor':'darkblue', 'font-weight':'bold'},
        style_data=dict(backgroundColor="dimgrey")
    )
    return table

# Callback for Portfolios Covariance Matric
@app.callback(Output('cov-matrix', 'children'),
              [Input('stockselector', 'value')])
def update_cov(selected_dropdown_value):
    sub_df = deepcopy(cov_matrix)
    sub_df = sub_df.round(6)
    table = dash_table.DataTable(
        id='cov_table',
        columns=[{"name": i, "id": i} 
                 for i in sub_df.columns],
        data=sub_df.to_dict('records'),
        style_cell={'textAlign':'left', 'font-family':'Open Sans Light'},
        style_header={'backgroundColor':'darkblue', 'font-weight':'bold'},
        style_data=dict(backgroundColor="dimgrey")
    )
    return table

# Callback for Portfolios Correlation Matrix
@app.callback(Output('corr-matrix', 'children'),
              [Input('stockselector', 'value')])
def update_corr(selected_dropdown_value):
    sub_df = deepcopy(corr_matrix)
    sub_df = sub_df.round(5)
    table = dash_table.DataTable(
        id='corr_table',
        columns=[{"name": i, "id": i} 
                 for i in sub_df.columns],
        data=sub_df.to_dict('records'),
        style_cell={'textAlign':'left', 'font-family':'Open Sans Light'},
        style_header={'backgroundColor':'darkblue', 'font-weight':'bold'},
        style_data=dict(backgroundColor="dimgrey")
    )
    return table

# Callback for Portfolios Matrix
@app.callback(Output('portfolios-matrix', 'children'),
              [Input('stockselector', 'value')])
def update_portfolios(selected_dropdown_value):
    sub_df = deepcopy(portfolios)
    sub_df = sub_df.sort_values(by=['Sharpe'], ascending=False)
    sub_df.rename(columns={'Sharpe': 'Sharpe Ratio'}, inplace=True)
    sub_df = sub_df.head(5)
    sub_df = sub_df.round(3)
    table = dash_table.DataTable(
        id='portfolios_table',
        columns=[{"name": i, "id": i} 
                 for i in sub_df.columns],
        data=sub_df.to_dict('records')  ,
        style_cell={'textAlign':'left', 'font-family':'Open Sans Light'},
        style_header={'backgroundColor':'darkblue', 'font-weight':'bold'},
        style_data=dict(backgroundColor="dimgrey")
    )
    return table

# Callback for News Analysis
@app.callback(Output('news', 'children'),
              [Input('stockselector', 'value')])
def update_news(selected_dropdown_value):
    sub_df = news[news['Ticker'] == selected_dropdown_value]
    sub_df = sub_df.drop(columns=['Ticker'])
    sub_df.rename(columns={'title':'Title', 'date':'Date', 'source':'Source'}, inplace=True)
    sub_df['Title'] = sub_df['Title'].apply(lambda row : row[:90])
    table = dash_table.DataTable(
        id='news_table',
        columns=[{"name": i, "id": i} 
                 for i in sub_df.columns],
        data=sub_df.to_dict('records'),
        style_cell={'textAlign':'left', 'font-family':'Open Sans Light'},
        style_header={'backgroundColor':'darkblue', 'font-weight':'bold'},
        style_data=dict(backgroundColor="dimgrey")
    )
    return table

# Callback for Copulas Frank Metrics
@app.callback(Output('copulas-table-frank', 'children'),
              [Input('stockselector', 'value')])
def update_frank_copulas(selected_dropdown_value):
    sub_df = copulas_table[copulas_table['Ticker'] == selected_dropdown_value]
    sub_df = sub_df[['Versus','Frank Kendall rank correlation', 'Frank spearmen correlation','Frank pearsons correlation','Frank theta of copula']]
    sub_df = sub_df.round(2)
    table = dash_table.DataTable(
        id='frank_table',
        columns=[{"name": i, "id": i} 
                 for i in sub_df.columns],
        data=sub_df.to_dict('records'),
        style_cell={'textAlign':'left', 'font-family':'Open Sans Light'},
        style_header={'backgroundColor':'darkblue', 'font-weight':'bold'},
        style_data=dict(backgroundColor="dimgrey")
    )
    return table

# Callback for Copulas Clayton Metrics
@app.callback(Output('copulas-table-clayton', 'children'),
              [Input('stockselector', 'value')])
def update_clayton_copulas(selected_dropdown_value):
    sub_df = copulas_table[copulas_table['Ticker'] == selected_dropdown_value]
    sub_df = sub_df[['Versus','Clayton Kendall rank correlation','Clayton spearmen correlation','Clayton pearsons correlation','Clayton theta of copula']]
    sub_df = sub_df.round(2)
    table = dash_table.DataTable(
        id='clayton_table',
        columns=[{"name": i, "id": i} 
                 for i in sub_df.columns],
        data=sub_df.to_dict('records'),
        style_cell={'textAlign':'left', 'font-family':'Open Sans Light'},
        style_header={'backgroundColor':'darkblue', 'font-weight':'bold'},
        style_data=dict(backgroundColor="dimgrey")
    )
    return table

# Callback for Copulas Gumbel Metrics
@app.callback(Output('copulas-table-gumbel', 'children'),
              [Input('stockselector', 'value')])
def update_gumbel_copulas(selected_dropdown_value):
    sub_df = copulas_table[copulas_table['Ticker'] == selected_dropdown_value]
    sub_df = sub_df[['Versus','Gumbel Kendall rank correlation','Gumbel spearmen correlation','Gumbel pearsons correlation','Gumbel theta of copula']]
    sub_df = sub_df.round(2)
    table = dash_table.DataTable(
        id='gumbel_table',
        columns=[{"name": i, "id": i} 
                 for i in sub_df.columns],
        data=sub_df.to_dict('records'),
        style_cell={'textAlign':'left', 'font-family':'Open Sans Light'},
        style_header={'backgroundColor':'darkblue', 'font-weight':'bold'},
        style_data=dict(backgroundColor="dimgrey")
    )
    return table

###################################################

# Callback for Monte-Carlo expected price in one year
@app.callback(Output('expected-price', 'children'),
              [Input('stockselector', 'value')])
def update_expected_price(selected_dropdown_value):
    sub_df = df_mc[df_mc['Ticker'] == selected_dropdown_value]
    result = sub_df['expected_price'].iloc[0]
    text = 'Expected Price in one year {} $.'.format(result)
    return text

# Callback for the Stock Analysis Title
@app.callback(Output('stock-title', 'children'),
              [Input('stockselector', 'value')])
def update_title(selected_dropdown_value):
    text = '{} Analysis'.format(selected_dropdown_value)
    return text

###################################################

if __name__ == '__main__':
    app.run_server(debug=True)
