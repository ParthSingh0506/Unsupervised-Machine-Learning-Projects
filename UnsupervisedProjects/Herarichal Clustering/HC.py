import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[: , [3,4]].values

#Using Dendogram to find the optimal number of clusters 

import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Using Dendogram For Finding Optimal number of cluster')
plt.xlabel('Customers')
plt.ylabel('Eucledian Distance')
plt.show()

#Triaing The Hierarichle Clustering model on the dataset

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(X)

#Visualsing The Clusters

plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1], s = 100 , c ='red' , label='Cluester1')
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1], s = 100 , c ='green' , label='Cluester2')
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1], s = 100 , c ='blue' , label='Cluester3')
plt.scatter(X[y_hc == 3,0],X[y_hc == 3,1], s = 100 , c ='orange' , label='Cluester4')
plt.scatter(X[y_hc == 4,0],X[y_hc == 4,1], s = 100 , c ='cyan' , label='Cluester5')
plt.title('Clusters on customers')
plt.xlabel('Anual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()