# -*- coding: utf-8 -*-
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, 3:].values

#Finding Optimum K(#Clusters)
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kMeans = KMeans(n_clusters = i, init = 'k-means++', n_init = 10, random_state = 0)
    kMeans.fit(x)
    wcss.append(kMeans.inertia_) #inertia is another name for wcss

#Plotting wcss to Visualize
plt.plot(range(1, 11), wcss)
plt.title('WCSS Visualization')
plt.xlabel('Clusters')
plt.ylabel('WCSS')
plt.show()

#Fitting to Data with optimum number of clusters
kMeans = KMeans(n_clusters = 5, init = 'k-means++', n_init = 10, random_state = 0)
cluster = kMeans.fit_predict(x)

plt.scatter(x[cluster == 0, 0], x[cluster == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
plt.scatter(x[cluster == 1, 0], x[cluster == 1, 1], s = 50, c = 'green', label = 'Cluster 2')
plt.scatter(x[cluster == 2, 0], x[cluster == 2, 1], s = 50, c = 'blue', label = 'Cluster 3')
plt.scatter(x[cluster == 3, 0], x[cluster == 3, 1], s = 50, c = 'yellow', label = 'Cluster 4')
plt.scatter(x[cluster == 4, 0], x[cluster == 4, 1], s = 50, c = 'pink', label = 'Cluster 5')
plt.scatter(kMeans.cluster_centers_[:, 0], kMeans.cluster_centers_[:, 1], s = 200, c = 'Black', label = 'Centroid')
plt.title('Clusters')
plt.xlabel('Income')
plt.ylabel('Score')
plt.legend()
plt.show()