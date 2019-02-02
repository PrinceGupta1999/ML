# -*- coding: utf-8 -*-
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting into TestSet and Training Set
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state = 0)

# Fitting SLR Model to Training Set(Take cares of Scaling)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xTrain, yTrain)

# Predicting Test Set Results
yPred = regressor.predict(xTest)

# Visualising Training Set
plt.scatter(xTrain, yTrain, color = 'red')
plt.plot(xTest, yPred)