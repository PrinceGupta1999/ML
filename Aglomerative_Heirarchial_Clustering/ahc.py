# -*- coding: utf-8 -*-
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, 3:].values

# Using Dendogram to Find Optimum No. of CLusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(x, method = 'ward')) # ward minimizes within cluster variance 

plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

#Fitting to Data with optimum number of clusters (5)
from sklearn.cluster import AgglomerativeClustering
ahc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
cluster = ahc.fit_predict(x)

#Visualizing the Clusters
plt.scatter(x[cluster == 0, 0], x[cluster == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
plt.scatter(x[cluster == 1, 0], x[cluster == 1, 1], s = 50, c = 'green', label = 'Cluster 2')
plt.scatter(x[cluster == 2, 0], x[cluster == 2, 1], s = 50, c = 'blue', label = 'Cluster 3')
plt.scatter(x[cluster == 3, 0], x[cluster == 3, 1], s = 50, c = 'yellow', label = 'Cluster 4')
plt.scatter(x[cluster == 4, 0], x[cluster == 4, 1], s = 50, c = 'pink', label = 'Cluster 5')
plt.title('Clusters')
plt.xlabel('Income')
plt.ylabel('Score')
plt.legend()
plt.show()