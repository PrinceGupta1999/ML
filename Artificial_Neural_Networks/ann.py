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
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
xTrain = scX.fit_transform(xTrain)
xTest = scX.transform(xTest)


# Imporiting Keras Libraries and Packages
from keras.models import Sequential
from keras.layers import Dense

# Initaializing ANN
classifier = Sequential()

# Adding Input and First Hidden Layers
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#                   # Neurons in layer, Initialize weights, Rectifier phi func, # Inputs to this Layer

# Adding 2nd Hidden Layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding O/P Layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#               function for updating weight, cost function, metric for evaluating performance 

# Fitting Model to Training Set
classifier.fit(xTrain, yTrain, batch_size = 10, epochs = 100)

# Predict Test Set Results
yPred = classifier.predict(xTest)
yPred = (yPred > 0.5)
# Make Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yTest, yPred)
