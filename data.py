import os
import math
import scipy
import pandas_datareader

import numpy as np
import pandas as pd
import yfinance as yf

from scipy import stats
from copy import deepcopy
from datetime import datetime
from GoogleNews import GoogleNews
from copulalib.copulalib import Copula
from yahoofinancials import YahooFinancials

###################################################

np.random.seed(123)

###################################################

def build_price_data(list_stocks):
    df = pd.DataFrame()
    for stock in list_stocks:
        tmp = yf.Ticker(stock).history(period='1y')
        tmp['Ticker'] = stock
        df = pd.concat([df, tmp])
    df['Return'] = (df.Open - df.Open.shift(1))/df.Open.shift(1)
    return df

def build_shareholders_data(list_stocks):
    df = pd.DataFrame()
    for stock in list_stocks:
        tmp = yf.Ticker(stock).institutional_holders
        tmp['Ticker'] = stock
        df = pd.concat([df, tmp])
    return df

def monte_carlo_expected(stock_name, number_of_trading_days = 252, number_of_trials = 5000):
    closing_prices = list()
    prices_trials = list()
    stock = pandas_datareader.data.DataReader(stock_name, 'yahoo', start='1/1/2010')
    time_elapsed = (stock.index[-1] - stock.index[0]).days
    total_growth = (stock['Adj Close'][-1] / stock['Adj Close'][1])
    number_of_years = time_elapsed / 365.0 
    cagr = total_growth ** (1/number_of_years) - 1
    std_dev = stock['Adj Close'].pct_change().std()
    std_dev = std_dev * math.sqrt(number_of_trading_days)
    daily_return_percentages = np.random.normal(cagr/number_of_trading_days, std_dev/math.sqrt(number_of_trading_days),number_of_trading_days)+1
    price_series = [stock['Adj Close'][-1]]
    for x in daily_return_percentages:
        price_series.append(price_series[-1] * x)
    for i in range(number_of_trials):
        daily_return_percentages = np.random.normal(cagr/number_of_trading_days, std_dev/math.sqrt(number_of_trading_days), number_of_trading_days)+1
        price_series = [stock['Adj Close'][-1]]  
        for j in daily_return_percentages:
            price_series.append(price_series[-1] * j)
        closing_prices.append(price_series[-1])
        prices_trials.append(price_series)  
    mean_end_price = round(np.mean(closing_prices),2)
    expected_price = str(mean_end_price)
    return expected_price, prices_trials, closing_prices

def build_mc_data(list_stocks):
    rows_list = list()
    for stock in list_stocks:
        expected_price, price_trials, closing_prices = monte_carlo_expected(stock)
        dict_stock = {'expected_price':expected_price, 'price_trials':price_trials, 'closing_prices':closing_prices, 'Ticker':stock}
        rows_list.append(dict_stock)
    df = pd.DataFrame(rows_list)
    return df


def build_esg_data(list_stocks):
    df = pd.DataFrame()
    list_values = ['socialScore', 'governanceScore', 'environmentScore', 'totalEsg', 'esgPerformance']
    for stock in list_stocks:
        tmp = yf.Ticker(stock).sustainability.reset_index()
        tmp = tmp[tmp['2021-5'].isin(list_values)]
        tmp['Ticker'] = stock
        df = pd.concat([df, tmp])
    return df

def build_additional_data(list_stocks):
    row_list = list()
    for stock in list_stocks:
        json = YahooFinancials(stock).get_historical_price_data("2020-05-22", "2021-05-22", "daily")
        df = pd.DataFrame(columns=["open","close","adjclose"])
        for row in json[stock]["prices"]:
            date = datetime.fromisoformat(row["formatted_date"])
            df.loc[date] = [row["open"], row["close"], row["adjclose"]]
        df.index.name = "date"
        Q = df["adjclose"].pct_change().dropna()
        normal = stats.probplot(Q, dist=stats.norm)
        tdf, tmean, tsigma = stats.t.fit(Q)
        student = stats.probplot(Q, dist=stats.t, sparams=(tdf, tmean, tsigma))
        dict_result = {'normal_x':normal[0][0].tolist(), 'normal_y':normal[0][1].tolist(), 'student_x':student[0][0].tolist(), 'student_y':student[0][1].tolist(), 'Ticker':stock}
        row_list.append(dict_result)
    result = pd.DataFrame(row_list)
    return result

def build_portfolio(list_stocks):
    
    df = pandas_datareader.data.DataReader(list_stocks, 'yahoo', start='2020/05/22', end='2021/05/23')
    df = df['Adj Close']
    cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()
    corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()
    ann_sd = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))

    #finding portfolio
    p_ret = [] # Define an empty array for portfolio returns
    p_vol = [] # Define an empty array for portfolio volatility
    p_weights = [] # Define an empty array for asset weights
    num_assets = len(df.columns)
    num_portfolios = 10000  
    ind_er = df.resample('Y').last().pct_change().mean()
    for portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights/np.sum(weights)
        p_weights.append(weights)
        returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its weights 
        p_ret.append(returns)
        var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum() # Portfolio Variance
        sd = np.sqrt(var) # Daily standard deviation
        ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
        p_vol.append(ann_sd)
    data = {'Returns':p_ret, 'Volatility':p_vol}
    for counter, symbol in enumerate(df.columns.tolist()):
        data[symbol+' weight'] = [w[counter] for w in p_weights]
    portfolios  = pd.DataFrame(data)
    portfolios['Sharpe'] = portfolios['Returns'] / portfolios['Volatility'] 
    return cov_matrix, corr_matrix, portfolios

def make_clickable(val):
    return '<a href="{}">{}</a>'.format(val,val)

def build_news_data(list_stocks):
    rows_list = list()
    for stock in list_stocks:
        googleNews = GoogleNews(period='7d')
        googleNews.get_news(stock)
        list_results = googleNews.results()
        for i in range(3):
            dict_result = list_results[i]
            dict_result['Ticker'] = stock
            rows_list.append(dict_result)
    df = pd.DataFrame(rows_list)
    df = df[['title', 'date', 'site', 'Ticker']]
    df.rename(columns={'site':'source'}, inplace=True)
    #df['link'] = df.apply(lambda x : x['link'], axis=1)
    return df

def get_copulas_data(stock1, stock2):
    df1 = pandas_datareader.data.DataReader([stock1], 'yahoo', start='2020/05/22', end='2021/05/23')
    df1 = df1['Adj Close']
    df2 = pandas_datareader.data.DataReader([stock2], 'yahoo', start='2020/05/22', end='2021/05/23')
    df2 = df2['Adj Close']
    x1=np.array(df1.pct_change().dropna())
    y1=np.array(df2.pct_change().dropna())
    x2=df1.pct_change().dropna().values
    y2=df2.pct_change().dropna().values
    x=x1.flatten()
    y=y1.flatten()
    y2=y2.reshape(-1)
    x2=x2.reshape(-1)
    returns=pd.DataFrame([x2,y2]).T
    returns.columns=['x','y']
    return x, y, returns

def build_copulas_data(list_stocks):
    row_list_plots = list()
    row_list_table = list()
    for stock in list_stocks:
        tmp_list = deepcopy(list_stocks)
        tmp_list.remove(stock)
        for versus in tmp_list:
            x, y, _ = get_copulas_data(stock, versus)
            frank = Copula(x,y,family='frank')
            uf,vf = frank.generate_uv(1000)
            clayton = Copula(x,y,family='clayton')
            uc,vc = clayton.generate_uv(1000)
            gumbel = Copula(x,y,family='gumbel')
            ug,vg = gumbel.generate_uv(1000)
            dict_result_plots = {'Ticker':stock, 'Versus':versus, 'uf':uf.tolist(), 'vf':vf.tolist(),
                                 'uc':uc.tolist(), 'vc':vc.tolist(), 'ug':ug.tolist(), 'vg':vg.tolist()}
            dict_result_table = {'Ticker':stock, 'Versus':versus, 'Frank Kendall rank correlation':frank.tau,'Frank spearmen correlation':frank.sr,
                                'Frank pearsons correlation':frank.pr,'Frank theta of copula':frank.theta,
                                'Clayton Kendall rank correlation':clayton.tau,'Clayton spearmen correlation':clayton.sr,
                                'Clayton pearsons correlation':clayton.pr,'Clayton theta of copula':clayton.theta,
                                'Gumbel Kendall rank correlation': gumbel.tau,'Gumbel spearmen correlation':gumbel.sr,
                                'Gumbel pearsons correlation':gumbel.pr,'Gumbel theta of copula':gumbel.theta}
            row_list_plots.append(dict_result_plots)
            row_list_table.append(dict_result_table)
    result_plots = pd.DataFrame(row_list_plots)
    result_table = pd.DataFrame(row_list_table)
    return result_plots, result_table
    

###################################################

list_stocks = ['AAPL', 'MSFT', 'TSLA', 'AMZN']
df_price = build_price_data(list_stocks)
df_shareholders = build_shareholders_data(list_stocks)
df_mc = build_mc_data(list_stocks)
df_esg = build_esg_data(list_stocks)
cov_matrix, corr_matrix, portfolios = build_portfolio(list_stocks)
df_additional = build_additional_data(list_stocks)
df_news = build_news_data(list_stocks)
df_copulas_plots, df_copulas_table = build_copulas_data(list_stocks)

###################################################

df_price.to_csv(os.getcwd() + '\\assets\\price.csv')
df_shareholders.to_csv(os.getcwd() + '\\assets\\shareholders.csv', index=False)
df_mc.to_csv(os.getcwd() + '\\assets\\monte-carlo.csv', index=False)
df_esg.to_csv(os.getcwd() + '\\assets\\esg.csv', index=False)
cov_matrix.to_csv(os.getcwd() + '\\assets\\cov_portfolio.csv')
corr_matrix.to_csv(os.getcwd() + '\\assets\\corr_portfolio.csv')
portfolios.to_csv(os.getcwd() + '\\assets\\portfolios.csv', index=False)
df_additional.to_csv(os.getcwd() + '\\assets\\additional.csv')
df_news.to_csv(os.getcwd() + '\\assets\\news.csv', index=False)
df_copulas_plots.to_csv(os.getcwd() + '\\assets\\copulas_plots.csv', index=False)
df_copulas_table.to_csv(os.getcwd() + '\\assets\\copulas_table.csv', index=False)