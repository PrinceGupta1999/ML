# -*- coding: utf-8 -*-
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(dataset.shape[0]):
    transactions.append([ str(dataset.values[i, j]) for j in range(dataset.shape[1]) ])

#Training Apriori on data 
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, 
                min_lift = 3, min_length = 2)

result = list(rules)