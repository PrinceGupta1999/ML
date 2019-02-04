# -*- coding: utf-8 -*-
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding Categorial Data
from sklearn.preprocessing import LabelEncoder
# for allowing priority of different categories (small, medium ..)
labelX = LabelEncoder()
x[:, -1] = labelX.fit_transform(x[:, -1])
from sklearn.preprocessing import OneHotEncoder
#for no priority among different categories
labelX = OneHotEncoder(categorical_features = [-1])
x = labelX.fit_transform(x).toarray()

# Avoiding Dummy Variable Trap (Removing First Column). However most libs take care of it auto
x = x[:, 1:]

# Splitting into TestSet and Training Set
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression Model (All-In Technique)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xTrain, yTrain)

# Predicting Test Set Results
yPred = regressor.predict(xTest)

# Building Optimal Model using Backward Elimination
# This library does not take care of constant itself and hence allows better control
import statsmodels.formula.api as sm 
x = np.append(np.ones((len(x), 1)).astype(int), values = x, axis = 1)

xOpt = x[:, [0, 1, 2, 3, 4, 5]]
regressorOLS = sm.OLS(endog = y, exog = xOpt).fit() # OLS = Ordinary Least Squares
regressorOLS.summary()

xOpt = x[:, [0, 1, 3, 4, 5]]
regressorOLS = sm.OLS(endog = y, exog = xOpt).fit() # OLS = Ordinary Least Squares
regressorOLS.summary()

xOpt = x[:, [0, 3, 4, 5]]
regressorOLS = sm.OLS(endog = y, exog = xOpt).fit() # OLS = Ordinary Least Squares
regressorOLS.summary()

xOpt = x[:, [0, 3, 5]]
regressorOLS = sm.OLS(endog = y, exog = xOpt).fit() # OLS = Ordinary Least Squares
regressorOLS.summary()

# Can Stop Here as P-value = 6%
xOpt = x[:, [0, 3]]
regressorOLS = sm.OLS(endog = y, exog = xOpt).fit() # OLS = Ordinary Least Squares
regressorOLS.summary()
