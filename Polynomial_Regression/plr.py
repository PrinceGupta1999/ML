# -*- coding: utf-8 -*-
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Fitting Aimple Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)

# Fitting Polynomial Linear Regression Model
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree = 4)
xPoly = polyReg.fit_transform(x)

regressor2 = LinearRegression()
regressor2.fit(xPoly, y)

# Visualizing Simple Linear Model
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Salary VS Experience')
plt.xlabel('Experience (Year)')
plt.ylabel('Salary ($)')
plt.show()

# Visualizing Polynomial Linear Model
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor2.predict(xPoly), color = 'blue')
plt.title('Salary VS Experience')
plt.xlabel('Experience (Year)')
plt.ylabel('Salary ($)')
plt.show()

# Lowering Step to get Continuous Curve
xGrid = np.arange(min(x), max(x), 0.1)
xGrid = xGrid.reshape(len(xGrid), 1)

# Visualizing Polynomial Linear Model
plt.scatter(x, y, color = 'red')
plt.plot(xGrid, regressor2.predict(polyReg.fit_transform(xGrid)), color = 'blue')
plt.title('Salary VS Experience')
plt.xlabel('Experience (Year)')
plt.ylabel('Salary ($)')
plt.show()
