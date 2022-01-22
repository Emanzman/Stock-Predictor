# Stock Prediction Program
#
# Author: Emmanuel Eyob

import quandl
import numpy as np  
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Getting stock data and Adj. Close Price

df = quandl.get("WIKI/FB")
df = df[["Adj. Close"]]

# No. of days forecasting into the future
future_forecast_val = int(input("How many future days do you want to find stock price predictions for?"))
forecase_out = future_forecast_val

# Column where Adj. Close values shifted up "forecast_out" times

df["Prediction"] = df[["Adj. Close"]].shift(-forecase_out)

# Convert the dataframe into a numpy array for easier data processing

X = np.array(df.drop(["Prediction"], 1))
X = X[:-forecase_out]


y = np.array(df["Prediction"])
y = y[:-forecase_out]

x_train, x_test, y_train, y_test = train_test_split(X, y)

# Training the Support Vector Machine

svr_rbf = SVR(kernel='rbf', C= 1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)

# Test Model where score returns score of confidence in prediction
# Best possible score = 1

svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence)

# Training Linear Regression Model

lr = LinearRegression()

lr.fit(x_train, y_train)

# Test Model where score returns score of confidence in prediction
# Best possible score = 1

lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)

# x_forecast = last 30 rows of original data set from Adj. Close column
x_forecast = np.array(df.drop(["Prediction"], 1))[-forecase_out:]

# Print Original Df
print(df

# Print linerar regression predictions for next days
lr_prediction = lr.predict(x_forecast)
print("Lr_prediction: ")
print(lr_prediction)

# Print support vector regressor model predictions for next days
svm_prediction = svr_rbf.predict(x_forecast)
print("Svm_ Prediction: ")
print(svm_prediction)