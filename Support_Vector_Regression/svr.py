# -*- coding: utf-8 -*-
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
scY = StandardScaler()
x = scX.fit_transform(x)
y = scY.fit_transform(y.reshape((len(y), 1)))
y = y[:, -1]

# Fitting SVR Model to the DataSet
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') # RBF is Gaussian
regressor.fit(x, y)

# Predicting New Value from Regressor
yPred = scY.inverse_transform(regressor.predict(scX.transform([[6.5]])))

# Visualizing SVR Model
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Salary VS Experience')
plt.xlabel('Experience (Year)')
plt.ylabel('Salary ($)')
plt.show()


# Lowering Step to get Continuous Curve
xGrid = np.arange(min(x), max(x), 0.1)
xGrid = xGrid.reshape(len(xGrid), 1)
plt.scatter(x, y, color = 'red')
plt.plot(xGrid, regressor.predict(xGrid), color = 'blue')
plt.title('Salary VS Experience')
plt.xlabel('Experience (Year)')
plt.ylabel('Salary ($)')
plt.show()
