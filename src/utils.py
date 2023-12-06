import pandas as pd
import random, os
import numpy as np
import tensorflow as tf
##########################################################################################################################
# prediction based equally weighted portfolios

def build_portfolios(ret_hat_capm, ret_hat_ff3, ret_hat_ff5, ret_hat_ex, weight='equal', test_data=None):
  df = pd.DataFrame()
  portfolios_ = pd.DataFrame()
  df['Size'] = test_data['Size']
  # actual returns
  df['ab_capm'] = test_data['ab_capm']/100
  df['ab_ff3'] = test_data['ab_ff3']/100
  df['ab_ff5'] = test_data['ab_ff5']/100
  df['ex_ret'] = test_data['ex_return']/100
  # predicted returns
  df['ab_capm_hat'] = ret_hat_capm
  df['ab_ff3_hat'] = ret_hat_ff3
  df['ab_ff5_hat'] = ret_hat_ff5
  df['ex_ret_hat'] = ret_hat_ex
  
  # equal weighted portfolios based on predictions
  if weight=='equal':
    df['capm_deciles'] = df.groupby('year_month')['ab_capm_hat'].transform(lambda g: pd.qcut(g, q=10, labels=list(f'Decile_{i}' for i in range(1,11))))
    df['ff3_deciles'] = df.groupby('year_month')['ab_ff3_hat'].transform(lambda g: pd.qcut(g, q=10, labels=list(f'Decile_{i}' for i in range(1,11))))
    df['ff5_deciles'] = df.groupby('year_month')['ab_ff5_hat'].transform(lambda g: pd.qcut(g, q=10, labels=list(f'Decile_{i}' for i in range(1,11))))
    df['ex_deciles'] = df.groupby('year_month')['ex_ret_hat'].transform(lambda g: pd.qcut(g, q=10, labels=list(f'Decile_{i}' for i in range(1,11))))
    
    portfolios_[['year_month', 'capm_deciles', 'ab_capm']] = df.groupby(['year_month', 'capm_deciles'])['ab_capm'].mean().reset_index()
    portfolios_[['year_month', 'ff3_deciles', 'ab_ff3']] = df.groupby(['year_month', 'ff3_deciles'])['ab_ff3'].mean().reset_index()
    portfolios_[['year_month', 'ff5_deciles', 'ab_ff5']] = df.groupby(['year_month', 'ff5_deciles'])['ab_ff5'].mean().reset_index()
    portfolios_[['year_month', 'ex_deciles', 'ex_ret']] = df.groupby(['year_month', 'ex_deciles'])['ex_ret'].mean().reset_index()
    return portfolios_
    
  # value weighted portfolios based on predictions
  if weight=='value':
    # build portfolios based on predicted capm abnormal returns
    df['capm_deciles'] = df.groupby('year_month')['ab_capm_hat'].transform(lambda g: pd.qcut(g, q=10, labels=list(f'Decile_{i}' for i in range(1,11))))
    # compute weightes
    df['size_sum'] = df.groupby(['year_month', 'capm_deciles'])['Size'].transform('sum')
    df['weight'] = df['Size']/df['size_sum']
    df['ab_capm'] = df['ab_capm']*df['weight']
    
    # build portfolios based on predicted ff3 abnormal returns
    df['ff3_deciles'] = df.groupby('year_month')['ab_ff3_hat'].transform(lambda g: pd.qcut(g, q=10, labels=list(f'Decile_{i}' for i in range(1,11))))
    df['size_sum'] = df.groupby(['year_month', 'ff3_deciles'])['Size'].transform('sum')
    df['weight'] = df['Size']/df['size_sum']
    df['ab_ff3'] = df['ab_ff3']*df['weight']
    
    # build portfolios based on predicted ff5 abnormal returns
    df['ff5_deciles'] = df.groupby('year_month')['ab_ff5_hat'].transform(lambda g: pd.qcut(g, q=10, labels=list(f'Decile_{i}' for i in range(1,11))))
    df['size_sum'] = df.groupby(['year_month', 'ff5_deciles'])['Size'].transform('sum')
    df['weight'] = df['Size']/df['size_sum']
    df['ab_ff5'] = df['ab_ff5']*df['weight']
    
    # build portfolios based on predicted excess returns
    df['ex_deciles'] = (df.groupby('year_month')['ex_ret_hat'].transform(lambda g: pd.qcut(g, q=10, labels=list(f'Decile_{i}' for i in range(1,11)))))
    df['size_sum'] = df.groupby(['year_month', 'ex_deciles'])['Size'].transform('sum')
    df['weight'] = df['Size']/df['size_sum']
    df['ex_ret'] = df['ex_ret']*df['weight']
    
    portfolios_[['year_month', 'capm_deciles', 'ab_capm']] = df.groupby(['year_month', 'capm_deciles'])['ab_capm'].sum().reset_index()
    portfolios_[['year_month', 'ff3_deciles', 'ab_ff3']] = df.groupby(['year_month', 'ff3_deciles'])['ab_ff3'].sum().reset_index()
    portfolios_[['year_month', 'ff5_deciles', 'ab_ff5']] = df.groupby(['year_month', 'ff5_deciles'])['ab_ff5'].sum().reset_index()
    portfolios_[['year_month', 'ex_deciles', 'ex_ret']] = df.groupby(['year_month', 'ex_deciles'])['ex_ret'].sum().reset_index()
    portfolios_.reset_index(inplace=True)
    return portfolios_

  else:
    print('Warning: such weighting method is not defined!')

##########################################################################################################################
def portfolio_cumulative_return(data):
  # compute cumulative return in each deciles
  portfolio_cum = pd.DataFrame()
  portfolio_cum['ab_capm'] = data.groupby('capm_deciles')['ab_capm'].cumsum()
  portfolio_cum['ab_ff3'] = data.groupby('ff3_deciles')['ab_ff3'].cumsum()
  portfolio_cum['ab_ff5'] = data.groupby('ff5_deciles')['ab_ff5'].cumsum()
  portfolio_cum['ex_ret'] = data.groupby('ex_deciles')['ex_ret'].cumsum()
  portfolio_cum['year_month'] = data['year_month'].astype(str)
  portfolio_cum['year_month'] = pd.to_datetime(portfolio_cum['year_month'], format='%Y-%m')
  portfolio_cum['capm_deciles'] = data['capm_deciles'].astype(str)
  portfolio_cum['ff3_deciles'] = data['ff3_deciles'].astype(str)
  portfolio_cum['ff5_deciles'] = data['ff5_deciles'].astype(str)
  portfolio_cum['ex_deciles'] = data['ex_deciles'].astype(str)
  
  return portfolio_cum

##########################################################################################################################
def prediction_long_short(data):
  long_short = pd.DataFrame()
  # long_short datas for capm abnormal return
  long = data[data['capm_deciles'] == 'Decile_10'][['year_month', 'ab_capm']]
  short = data[data['capm_deciles'] == 'Decile_1'][['year_month', 'ab_capm']]
  # format date
  long_short.index = long['year_month'].astype(str)
  long_short.index = pd.to_datetime(long_short.index, format='%Y-%m')
  long_short['ab_capm'] = long['ab_capm'].values.reshape(-1,1) - short['ab_capm'].values.reshape(-1,1)
  
  # long-short datas for ff3 abnormal return
  long = data[data['ff3_deciles'] == 'Decile_10'][['year_month', 'ab_ff3']]
  short = data[data['ff3_deciles'] == 'Decile_1'][['year_month', 'ab_ff3']]
  long_short['ab_ff3'] = long['ab_ff3'].values.reshape(-1,1) - short['ab_ff3'].values.reshape(-1,1)
  
  # long-short datas for ff5 abnormal return
  long = data[data['ff5_deciles'] == 'Decile_10'][['year_month', 'ab_ff5']]
  short = data[data['ff5_deciles'] == 'Decile_1'][['year_month', 'ab_ff5']]
  long_short['ab_ff5'] = long['ab_ff5'].values.reshape(-1,1) - short['ab_ff5'].values.reshape(-1,1)
  
  # long-short datas for ff5 abnormal return
  long = data[data['ex_deciles'] == 'Decile_10'][['year_month', 'ex_ret']]
  short = data[data['ex_deciles'] == 'Decile_1'][['year_month', 'ex_ret']]
  long_short['ex_ret'] = long['ex_ret'].values.reshape(-1,1) - short['ex_ret'].values.reshape(-1,1)

  return long_short

#########################################################################################################################
# set random seed for reproducibility
def set_seed(SEED):
  os.environ['PYTHONHASHSEED']=str(SEED)
  np.random.seed(SEED)
  tf.random.set_seed(SEED)
  random.seed(SEED)
  
