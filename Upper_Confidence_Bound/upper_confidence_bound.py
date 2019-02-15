# -*- coding: utf-8 -*-
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implement UCB
d = dataset.shape[1]
N = dataset.shape[0]
# Step 1
selection = [0] * d
reward = [0] * d
selectedAds = []

from math import sqrt, log
for n in range(N):
    bestUpperBound = 0
    ad = 0
    for i in range(d):
        # if else to choose each add once for 1st d loops 
        if selection[i] > 0:
            avgReward = reward[i] / selection[i]
            delta = sqrt((3 / 2) * (log(n + 1) / selection[i]))
            upperBound = avgReward + delta
        else:
            upperBound = 1e400
            
        if upperBound > bestUpperBound:
            ad = i
            bestUpperBound = upperBound
    selectedAds.append(ad)
    selection[ad] += 1
    reward[ad] += dataset.values[n, ad]
totalReward = sum(reward)

#Visualising Result
plt.hist(selectedAds)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
