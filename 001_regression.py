import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import pickle

import datetime
# import matplotlib.pyplot as plt
# from matplotlib import style

# style.use('ggplot')

## Get the data
df = quandl.get('WIKI/GOOGL')

## Add calculated columns to dataset
df['HL_PCT'] = ((df['Adj. High'] - df['Adj. Close'])/df['Adj. Close']) * 100
df['PCT_change'] = ((df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open']) * 100

## Create a new dataframe with FEATURES
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

## Create a column with LABEL

# Initialize with Adj.close data
forecast_col = 'Adj. Close'

# Fill the NA with rediculous high number
# to not lose the row during df.dropna
df.fillna(-99999, inplace=True)

# shift the value by 3 days
forecast_out = 3
df['label'] = df[forecast_col].shift(-forecast_out)

## Drop the columns with NA
## the last 3 rows will be deleted because
## the label has NaN
# df.dropna(inplace=True)

## x = Features
## Take all the rows without column "label"
x = np.array(df.drop(['label'], 1))
x = preprocessing.scale(x)
## Get the rows to forecast for next n days
x_lately = x[-forecast_out:]
## Get the rows without the forcast rows
x = x[:-forecast_out]

## y = Label
df.dropna(inplace=True)
y = np.array(df['label'])

## make sure they are of same length
print(len(x), len(y))

## Cross validation
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.2)

## Using a particular model
classifier = LinearRegression()
classifier.fit(x_train, y_train)
with open('linearregression.pickle', 'wb') as fh:
    pickle.dump(classifier, fh)

classifier = pickle.load(open('linearregression.pickle', 'rb'))

accuracy = classifier.score(x_test, y_test)
print(accuracy)

## Prediction
forecast_set = classifier.predict(x_lately)
print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

## get the last date in the dataframe, iloc is locate
last_date = df.iloc[-1].name
last_timestamp = last_date.timestamp()
one_day = 86400 ## secs in a day
next_timestamp = last_timestamp + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_timestamp)
    next_timestamp += one_day
    ## locate if present, update it else add it
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

# df['Adj. Close'].plot()
# df['Forecast'].plot()
# plt.legend(loc=4)
# plt.xlabel('Date')
# plt.ylable('Price')
# plt.show()

print(df.tail())
