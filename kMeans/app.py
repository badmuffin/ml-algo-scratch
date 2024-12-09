from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from kmeans import KMeans
# import pandas as pd

centroids = [(-5, -5), (5, 5), (-2.5, 2.5), (2.5, -2.5)]
cluster_std = [1, 1, 1, 1] #cluster standara deviation
X, Y = make_blobs(n_samples=100, cluster_std=cluster_std, centers=centroids, n_features=2, random_state=2)

#df = pd.read_csv('your_data.csv')
# X = df.iloc[:, :].values

#we need to specify maximum iteration for better accuracy
km = KMeans(n_clusters=4, max_iter=200) 
Y_means = km.fit_predict(X);

#first cluster
plt.scatter(X[Y_means == 0, 0], X[Y_means == 0, 1], color='red')
#second cluster
plt.scatter(X[Y_means == 1, 0], X[Y_means == 1, 1], color='blue')
#third cluster
plt.scatter(X[Y_means == 2, 0], X[Y_means == 2, 1], color='green')
#fourth cluster
plt.scatter(X[Y_means == 3, 0], X[Y_means == 3, 1], color='orange')


plt.show();
