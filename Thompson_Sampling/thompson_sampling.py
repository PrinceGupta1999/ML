# -*- coding: utf-8 -*-
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implement Thompson Sampling
d = dataset.shape[1]
N = dataset.shape[0]
# Step 1
rewardOne = [0] * d
rewardZero = [0] * d
selectedAds = []

import random
for n in range(N):
    mxRandomVal = 0
    ad = 0
    for i in range(d):
            randomBeta = random.betavariate(rewardOne[i] + 1, rewardZero[i] + 1)
            if randomBeta > mxRandomVal:
                mxRandomVal = randomBeta
                ad = i
    selectedAds.append(ad)
    if dataset.values[n, ad] == 1:
        rewardOne[ad] += 1
    rewardZero[ad] += 1
totalReward = sum(rewardOne)

#Visualising Result
plt.hist(selectedAds)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
