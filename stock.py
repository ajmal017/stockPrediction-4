import math
import pandas as pd
import numpy as np
import datetime

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from pandas_datareader import data as pdr
import fix_yahoo_finance as yf

import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl

dataframe = pdr.get_data_yahoo('INFY', start='2018-01-01').reset_index()
dfreg = dataframe.loc[:, ['Date', 'Volume', 'Open', 'High', 'Low', 'Close', 'Adj Close']]

dfreg['Date'] = pd.to_datetime(dfreg['Date'], format='%Y-%m-%d')
dfreg = dfreg.sort_values(by=['Date'], ascending=[True])
dfreg.set_index('Date', inplace=True)
dfreg = dfreg.resample('D').fillna(method=None).interpolate()

dfreg.head()


dfreg['HL_PCT'] = (dataframe['High'] - dataframe['Low']) / dataframe['Close'] * 100.0
dfreg['PCT_change'] = (dataframe['Close'] - dataframe['Open']) / dataframe['Open'] * 100.0

# Drop missing value
dfreg.fillna(value=-99999, inplace=True)
# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))
# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))
# Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)
# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]

X = X[:-forecast_out]
# Separate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]
y_lately = y[-forecast_out:]

# So, how'd we do?

# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X, y)
clfreg_confidence = clfreg.score(X, y) * 100
print('Linear regression score: {:.2f}%'.format(clfreg_confidence))


clfridge = Ridge(alpha=1.0)
clfridge.fit(X, y)
clfridge_confidence = clfridge.score(X, y) * 100
print('Ridge regression score: {:.2f}%'.format(clfridge_confidence))


clflasso = Lasso(alpha=0.1)
clflasso.fit(X, y)
clflasso_confidence = clflasso.score(X, y) * 100
print('Lasso regression score: {:.2f}%'.format(clflasso_confidence))

# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X, y)
clfpoly2_confidence = clfpoly2.score(X, y) * 100
print('Quadratic Regression 2 score: {:.2f}%'.format(clfpoly2_confidence))


print("Enough with the code. Let's get that cheddar! 💰🧀")


clfreg_forecast = clfreg.predict(X_lately)
clfridge_forecast = clfridge.predict(X_lately)
clflasso_forecast = clflasso.predict(X_lately)
clfpoly2_forecast = clfpoly2.predict(X_lately)
forecast_set = (clfreg_forecast + clfridge_forecast + clflasso_forecast + clfpoly2_forecast) / 4

dfreg['Forecast'] = np.nan
last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in
                         range(len(dfreg.columns)-1)]+[i]

dfreg['Adj Close'].tail(100).plot()
dfreg['Forecast'].tail(100).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
