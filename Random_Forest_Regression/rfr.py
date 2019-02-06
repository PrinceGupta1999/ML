# -*- coding: utf-8 -*-
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Fitting Regression Model to the DataSet
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(x, y)

# Predicting New Value from Regressor
yPred = regressor.predict([[6.5]])

# Lowering Step to get Continuous Curve
xGrid = np.arange(min(x), max(x), 0.01)
xGrid = xGrid.reshape(len(xGrid), 1)
plt.scatter(x, y, color = 'red')
plt.plot(xGrid, regressor.predict(xGrid), color = 'blue')
plt.title('Salary VS Experience')
plt.xlabel('Experience (Year)')
plt.ylabel('Salary ($)')
plt.show()
