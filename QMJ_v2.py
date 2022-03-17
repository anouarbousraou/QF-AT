# import packages
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from datetime import datetime
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")

"""
Read data & add lagged variables
"""

# montly data
monthly_data = pd.read_csv('NL_FR_BE_data_monthly.csv', index_col=1)
monthly_data.index = pd.to_datetime(monthly_data.index, format='%Y%m')
monthly_data['lagged_ME'] = monthly_data.groupby('ISIN')['ME'].shift(1)
monthly_data['lagged_RET11'] = monthly_data.groupby('ISIN')['RET11'].shift(1)
monthly_data['lagged_ivol'] = monthly_data.groupby('ISIN')['ivol'].shift(1)
monthly_data['lagged_s'] = monthly_data.groupby('ISIN')['s'].shift(1)
monthly_data['fyear'] = monthly_data.index.year
monthly_data = monthly_data[['ISIN', 'RET', 'lagged_RET11', 'lagged_s', 'lagged_ME', 'lagged_ivol', 'fyear']]
monthly_data = monthly_data.sort_index()

# firm data
firms = pd.read_csv('NL_FR_BE_firms.csv')

# Fama & French factors
ff_factors = pd.read_csv('Europe_FF_Factors.csv', index_col=0)
ff_factors.index = pd.to_datetime(ff_factors.index, format='%Y%m')

# annual data
annual_data = pd.read_csv('NL_FR_BE_data_annual.csv', index_col=1)
# one lag for periods after JUNE
annual_data['lagged_OP_1'] = annual_data.groupby('ISIN')['OP'].shift(1)
annual_data['lagged_INV_1'] = annual_data.groupby('ISIN')['INV'].shift(1)
annual_data['lagged_BEME_1'] = annual_data.groupby('ISIN')['BEME'].shift(1)
# two lag, for periods before JUNE
annual_data['lagged_OP_2'] = annual_data.groupby('ISIN')['OP'].shift(2)
annual_data['lagged_INV_2'] = annual_data.groupby('ISIN')['INV'].shift(2)
annual_data['lagged_BEME_2'] = annual_data.groupby('ISIN')['BEME'].shift(2)
annual_data = annual_data[
    ['ISIN', 
    'lagged_OP_1', 
    'lagged_INV_1', 
    'lagged_BEME_1', 
    'lagged_OP_2', 
    'lagged_INV_2', 
    'lagged_BEME_2']
    ]

# merge monthly with annual data
monthly_data = monthly_data.reset_index()
unique_dates = list(monthly_data['mdate'])
unique_dates = set(unique_dates)
merged_data = pd.merge(monthly_data, annual_data, on=['ISIN', 'fyear'])
merged_data = merged_data.set_index('mdate')
merged_data = merged_data.dropna()
merged_data = merged_data.sort_index()

# for the periods after June, we can use the annual data from 1 year before, instead of 2 years before
# logic behind code -> np.where, month is before or equal to June, value if true, value if false
merged_data['month'] = merged_data.index.month
merged_data['lagged_OP'] = np.where(merged_data['month']<=6, merged_data.pop('lagged_OP_2'), merged_data.pop('lagged_OP_1'))
merged_data['lagged_INV'] = np.where(merged_data['month']<=6, merged_data.pop('lagged_INV_2'), merged_data.pop('lagged_INV_1'))
merged_data['lagged_BEME'] = np.where(merged_data['month']<=6, merged_data.pop('lagged_BEME_2'), merged_data.pop('lagged_BEME_1'))

"""
We define quantiles for each variable we base our strategy on
"""
# define cuts
n = 10
# create quantiles based on cuts
merged_data['OP_g'] = merged_data.groupby('mdate')['lagged_OP'].apply(lambda x: pd.qcut(x, n, labels=range(0, n)))
merged_data['ME_g'] = merged_data.groupby('mdate')['lagged_ME'].apply(lambda x: pd.qcut(x, n, labels=range(0, n)))
merged_data['s_g'] = merged_data.groupby('mdate')['lagged_s'].apply(lambda x: pd.qcut(x, n, labels=range(0, n)))
merged_data['RET11_g'] = merged_data.groupby('mdate')['lagged_RET11'].apply(lambda x: pd.qcut(x, 10, labels=range(0, 10)))

# define function which indentifies long(p01), neutral(p02) & short(p03) portfolio
def f(row):
    if (row['OP_g'] >= 8 and row['ME_g'] < 1 and row['s_g'] >= 8):
        val = 'p01'
    elif row['OP_g'] < 1 and row['ME_g'] >= 8 and row['s_g'] < 1:
        val = 'p03'
    else:
        val = 'p02'
    return val

# apply function
merged_data['qmjport'] = merged_data.apply(f, axis=1)

# identify companies we trade and frequency
long_companies = merged_data[merged_data['qmjport'] == 'p01']
long_companies['freq'] = long_companies.groupby('ISIN')['ISIN'].transform('count')
long_companies = long_companies.reset_index()
long_companies = long_companies.drop_duplicates(subset=['ISIN'])
long_companies = long_companies[['ISIN', 'freq']]
long_companies = long_companies.sort_values(by=['freq'], ascending=False)
long_companies = pd.merge(long_companies, firms, on='ISIN')

merged_data = merged_data.reset_index()
# make weights based on idio syncratic risk, lower risk gets more weight hence ascending is false
merged_data['rank'] = merged_data.groupby(['mdate', 'qmjport'])['lagged_ivol'].rank(method='dense', ascending=False)
merged_data['sum_rank'] = merged_data.groupby(['qmjport', 'mdate'])['rank'].transform('sum')
merged_data['weight'] = merged_data['rank']/merged_data['sum_rank']
merged_data['w*RET'] = merged_data['weight'] * merged_data['RET']

# #check if weights sum to 1, uncomment first...
# print(merged_data.groupby(['mdate', 'qmjport'])['weight'].sum())

p = ['p01', 'p03']
unique_data = merged_data[merged_data['qmjport'].isin(p)]

no_trade = 0
for i in unique_dates:
    if i not in unique_data['mdate'].values:
        print(f' Month where we dont find a good trade {i}')
        no_trade += 1

print(f' In total there are {no_trade} months where we did not make a trade')   

unique_companies = set(list(unique_data['ISIN']))
print(f' We have traded {len(unique_companies)} unique companies over the entire sample period')

# idiosyncratic risk weighting, lowest idio risk get more weight
qmj_portfolio = merged_data.groupby(['mdate','qmjport'])['w*RET'].sum()
qmj_portfolio = qmj_portfolio.reset_index()


# pivot the data
qmj_portfolio = qmj_portfolio.pivot(index = 'mdate', 
                                      columns = 'qmjport', 
                                      values = 'w*RET')
# if we didnt find a company to short, we replace NaN with 0                               
qmj_portfolio['p03'] = qmj_portfolio['p03'].replace(np.nan, 0)
qmj_portfolio = qmj_portfolio.dropna()
# long minus short return
qmj_portfolio['RET'] = qmj_portfolio['p01'] - qmj_portfolio['p03']
# merged with ff factors to get excess return
qmj_portfolio = pd.merge(qmj_portfolio, ff_factors, on='mdate')
qmj_portfolio['EXRET'] = qmj_portfolio['RET'] - qmj_portfolio['RF']

"""
Now we analyze
"""

# regression
x = qmj_portfolio[["MktRF","SMB","HML"]]
x = sm.add_constant(x)
model = sm.OLS(qmj_portfolio['EXRET'],x).fit(cov_type='HAC', cov_kwds={'maxlags':11})
print(model.summary())
print(f'*annualized alpha of {model.params[0]*12}')
resid = model.resid
# info & sharpe
info_ratio = model.params[0]/resid.std()
print(f'*annualized information ratio obtained is : {info_ratio*(12**0.5)}')
sharpe_ratio = np.mean(qmj_portfolio['p01']/np.std(qmj_portfolio['p01']))
print(f'*annualized sharpe ratio: {sharpe_ratio*(12**0.5)}')

# distribution & characteristics
ann_volatility = qmj_portfolio['RET'].std()*12**.5
print('***Annual volatility***', ann_volatility, sep='\n')
skew = stats.skew(qmj_portfolio['RET'])
print('***Skewness***', skew, sep='\n')
kur = stats.kurtosis(qmj_portfolio['RET'], fisher=True)
print('***Excess Kurtosis***', kur, sep='\n')

# create cumulative returns
cum_prod_port = np.cumprod(1 + qmj_portfolio['EXRET'])
cum_prod_mkt = np.cumprod(1 + qmj_portfolio['MktRF'])
cum_prod_smb = np.cumprod(1 + qmj_portfolio['SMB'])
cum_prod_hml = np.cumprod(1 + qmj_portfolio['HML'])
cum_prod_rf = np.cumprod(1 + qmj_portfolio['RF'])
cum_prod_wml = np.cumprod(1 + qmj_portfolio['WML'])

# plot excess returns
plt.figure(figsize=(16,5))
plt.subplot(1, 2, 1)
plt.plot(qmj_portfolio.index, qmj_portfolio['EXRET'], color='orange', marker='*')
plt.plot(qmj_portfolio.index, qmj_portfolio['MktRF'], color='blue', marker='^', alpha=0.7)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12, rotation=0)
plt.grid(axis='both', alpha=.3)
plt.gca().spines["top"].set_alpha(0.0)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.0)    
plt.gca().spines["left"].set_alpha(0.3)
plt.title('Plot of QMJ portfolio vs market')
plt.xlabel('Time', fontsize = 13, labelpad = 15)
plt.ylabel('Excess return', fontsize = 13, labelpad = 15)
plt.legend(['Portfolio excess return', 'Market excess return'])

# plot cumulative returns
plt.subplot(1, 2, 2)
plt.plot(cum_prod_port.index, cum_prod_port.values, color='orange')
plt.plot(cum_prod_mkt.index, cum_prod_mkt.values, color='purple')
plt.plot(cum_prod_smb.index, cum_prod_smb.values, color='green')
plt.plot(cum_prod_hml.index, cum_prod_hml.values, color='blue')
plt.plot(cum_prod_rf.index, cum_prod_rf.values, color='yellow')
plt.plot(cum_prod_wml.index, cum_prod_wml.values, color='brown')
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12, rotation=0)
plt.grid(axis='both', alpha=.3)
plt.gca().spines["top"].set_alpha(0.0)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.0)    
plt.gca().spines["left"].set_alpha(0.3)
plt.title('Cumulative excess return')
plt.legend(['QMJ Portfolio', 'Market', 'SMB', 'HML', 'RF', 'WML'])
plt.xlabel('Time', fontsize = 13, labelpad = 15)
plt.ylabel('Cumulative return', fontsize = 13, labelpad = 15)
plt.yscale('log')
plt.show()

#plot distribution
plt.subplot(1, 2, 1)
sns.barplot(data=long_companies.head(5), x="name", y="freq")
plt.title('Distribution top 5 traded companies', fontsize= 24)
plt.xticks(fontsize=9)
plt.yticks(fontsize=15)
plt.xlabel('Company', fontsize = 16)
plt.ylabel('Frequency traded', fontsize = 16)
plt.subplot(1, 2, 2)
sns.barplot(data=long_companies, x="country", y="freq")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Countries', fontsize = 18)
plt.ylabel('Distribution', fontsize = 16, labelpad = 15)
plt.title('Distribution countries', fontsize= 24)
plt.show()
