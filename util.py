import numpy as np
import pandas as pd
import math
import datetime
from ta import *
import quandl as Quandl
Quandl.ApiConfig.api_key = "msKYtye1W9DWW4CuVSnW"
import numpy as np
import pandas as pd
import math
from sklearn import preprocessing, cross_validation, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

##############################
######## GETTING DATA ########
##############################

def fetch_prices(ticker_list, limit=None):
    '''
    takes in ticker list of strings and pulls in their data from quandl
    can limit number of stocks to query with limit param
    '''
    if limit:
        ticker_list = ticker_list[:limit]

    out_dict = {}
    for ticker_str in ticker_list:
    #     print(ticker_str)
        try:
            ticker_df = Quandl.get(ticker_str,api_key="msKYtye1W9DWW4CuVSnW")
            out_dict[ticker_str] = ticker_df
        except:
            pass
            # print(ticker_str + " does not exist on Quandl's side.")
    return out_dict








############################################
####### CLEANING DATA & FEATURIZING ########
############################################

########## BATCHING ##########
def apply_to_all_stocks(function, stock_dict):
    '''given a dictionary (k,v)=(ticker,data) and a function,
        map the function onto each stock in the dictionary'''
    for stock_data,stock_ticker in zip(stock_dict.values(), stock_dict.keys()):
        stock_dict[stock_ticker] = function(stock_data)

    return stock_dict

def select_relevant_raw_features(df):
    # Pick the needed columns
    df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
    return df


def add_ft_PCT_change(df):
    # Create new features (percent change)
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

    return df


def add_industry_level_fts(df):
    # Create new features (percent change)
    df['HL_PCT'] = (df['High'] - df['Low']) / df['Index Value'] * 100.0
    df['MKT_PCT'] = (df['Index Value']  / df['Total Market Value'] * 100.0)

    return df


def add_market_level_fts(df):
    # Create new features (percent change)
    df['HL_PCT'] = (df['High'] - df['Low']) / df['Index Value'] * 100.0

    return df

def add_market_urc_level_fts(df):
    '''note nothing to be done rn so no code here'''
    # Create new features (percent change)
    #df['HL_PCT'] = (df['High'] - df['Low']) / df['Index Value'] * 100.0

    return df


#helper
def normalize_df(X):
    X = preprocessing.scale(X)
    return X

#helper
def fill_NaNs(df, value_to_fill=-99999):
    df = df.fillna(value=value_to_fill, inplace=False)
    return df


def clean_and_split(df, forecast_pct=0.05, forecast_col='Adj. Close', nan_fill_fn=fill_NaNs, normalizer_fn=normalize_df):
    '''prep df for train test using starter featurization'''

    #FILL NANS
    df = nan_fill_fn(df)
    df = df.replace([np.inf, -np.inf], np.nan, inplace=False)

    #SPLIT X AND Y
    forecast_out = int(math.ceil(forecast_pct * len(df)))
    df['label'] = df[forecast_col].shift(-forecast_out)
    df = df[:-forecast_out] #forecast until when we can look forward by n steps
    #nans = lambda df: df[df.isnull().any(axis=1)]
    #print(nans(df))
    #print("hello")
    df = nan_fill_fn(df)
    df = df.replace([np.inf, -np.inf], np.nan, inplace=False)

    X = np.array(df.drop(['label'], 1))


    # Only use the training set, and leave some values to predict on
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    #NORMALIZE
    X = normalizer_fn(X)

    y = np.array(df['label'])
    y = y[:-forecast_out]

    return X,y,X_lately




### Single stock indicators util functions
def select_indicators(single_stock_df):
    df = single_stock_df
    high = df['Adj. High']
    low = df['Adj. Low']
    close = df['Adj. Close']

    df['avg_directional'] = trend.adx(high, low, close, n=14, fillna=False)
    df['rsi'] = momentum.rsi(close, n=14, fillna=False)
    df['wr_pct'] = momentum.wr(high, low, close, lbp=14, fillna=False)
    df['volatility_avg'] = volatility.average_true_range(high, low, close, n=14, fillna=False)
    df['bband'] = volatility.bollinger_hband(close, n=20, ndev=2, fillna=False)
    df['bbandH'] = volatility.bollinger_hband(close, n=20, ndev=2, fillna=False)
    df['bbandL'] = volatility.bollinger_lband(close, n=20, ndev=2, fillna=False)
    df['bbandEMA'] = volatility.bollinger_mavg(close, n=20, fillna=False)

    return df


def drop_col(df, colname):
    '''drop the provided df col inplace'''
    df.drop(labels=[colname], axis=1, inplace=True)


def prune_date_range(full_df, restricted_df):
    '''assume you have two dfs with date indices and would like to filter the rows on the larger (full) df to those rows whose dates have matching rows in the smaller (restricted) df. this function returns the larger df, filtered as described'''

    pruned_df = full_df.loc[full_df.index.isin(restricted_df.index)]
    return pruned_df






##############################
####### MODELING FNS #########
##############################

def forward_chaining_train_test_split(X, y, k, min_train_pct, val_set_pct):
    '''
    split up data for cross-validation for time series!

    make k folds, uniformly spaced within the range.

    input: data matrix
    output: k-dim (for each fold) array of tuples (X_train, X_test, y_train, y_test)

    Example (k=5, val_set_pct is low s.t. val_set_size is 1, min_train_pct is low s.t. min_train_size = 1)
    fold 1 : training [1], test [2]
    fold 2 : training [1 2], test [3]
    fold 3 : training [1 2 3], test [4]
    fold 4 : training [1 2 3 4], test [5]
    fold 5 : training [1 2 3 4 5], test [6]

    from https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection
    '''
    min_train_idx = int(len(X) * min_train_pct)
    val_set_size = int(val_set_pct * len(X))
    step_size = ((len(X) - val_set_size) - min_train_idx) / (k-1)
    end_idx = len(X)

    folds = []
    for idx in np.arange(min_train_idx, end_idx-val_set_size, step=step_size):
        X_train, X_test, y_train, y_test = X[:idx], X[idx:idx+val_set_size], y[:idx], y[idx:idx+val_set_size]
        folds.append((X_train, X_test, y_train, y_test))

    return folds



def linear_regression(X, y):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    # Fit a linear regression model to the data
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(confidence)


def average_metrics(df, df_aug, clf):
    ''''''
    conf_aug, conf = [],[]
    for i in range(len(df)):
        X,y,_ = list(df.values())[i]
        X_aug,y_aug,_ = list(df_aug.values())[i]
        curr_conf, curr_conf_aug = run_model(X,y,clf), run_model(X_aug,y_aug,clf)
        #if i%20==0:
            #print(np.mean(conf), np.mean(conf_aug))
        conf_aug += [curr_conf_aug]
        conf+= [curr_conf]

    return np.mean(conf), np.mean(conf_aug)
# Specific models

def create_linear_models():
    models = {}
    models['linear_reg'] = LinearRegression(n_jobs=-1)
    models['huber_robust_linear'] = HuberRegressor()
    models['bayesian_ridge'] = BayesianRidge()

    return models

def run_model(X, y, clf):
    '''does crossval!'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)

    return confidence

def hmm_predict(X):
    diff = close_v[1:] - close_v[:-1]
    X = np.column_stack([diff, X])
    model = GaussianHMM(4, covariance_type="diag", n_iter=1000)
    model.fit([X])
    hidden_states = model.predict(X)

    print("Transition matrix:")
    print(model.transmat_)

    print("\nMeans and vars of each hidden state")
    for i in range(n_components):
        print("%dth hidden state" % i)
        print("mean = ", model.means_[i])
        print("var = ", np.diag(model.covars_[i]))


