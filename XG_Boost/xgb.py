# -*- coding: utf-8 -*-
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing Dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding Categorial Data
from sklearn.preprocessing import LabelEncoder
# For allowing priority of different categories (small, medium ..)
labelX = LabelEncoder()
x[:, 1] = labelX.fit_transform(x[:, 1])
labelX = LabelEncoder()
x[:, 2] = labelX.fit_transform(x[:, 2])
from sklearn.preprocessing import OneHotEncoder
# For no priority among different categories
labelX = OneHotEncoder(categorical_features = [1])
x = labelX.fit_transform(x).toarray()
# Avoiding Dummy Var Trap
x = x[:, 1:]

# Splitting into TestSet and Training Set
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(xTrain, yTrain)

yPred = classifier.predict(xTest)

# Make Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yTest, yPred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = xTrain, y = yTrain, cv = 10)
mean = accuracies.mean()
standardDeviation = accuracies.std()