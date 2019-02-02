# -*- coding: utf-8 -*-
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(np.nan, 'mean')
imputer = imputer.fit(x[:, 1:-1])
x[:, 1:-1] = imputer.transform(x[:, 1:-1])

# Encoding Categorial Data
from sklearn.preprocessing import LabelEncoder
#for allowing priority of different categories (small, medium ..)
labelX = LabelEncoder()
x[:, 0] = labelX.fit_transform(x[:, 0])
from sklearn.preprocessing import OneHotEncoder
#for no priority among different categories
labelX = OneHotEncoder(categorical_features = [0])
x = labelX.fit_transform(x).toarray()

# Dependent
labelY = LabelEncoder()
y = labelY.fit_transform(y)

# Splitting into TestSet and Training Set
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
xTrain = scX.fit_transform(xTrain)
xTest = scX.transform(xTest)
