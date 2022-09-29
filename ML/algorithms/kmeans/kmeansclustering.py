"""
From scratch implementation of K means clustering which is a unsupervised 
clustering  method that works by iteratively computing new centroids and 
moving centroids to the center of the new formed clusters.

Resource to learn more about K-means clustering by StatQuest:
https://www.youtube.com/watch?v=4b5d3muPQmA

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-05-28 Initial coding
*    2022-09-29 Ensured code works as intended and added docstrings
"""

# Imports
import numpy as np  # For matrix operations
import matplotlib.pyplot as plt  # For plotting
from sklearn.datasets import make_blobs  # For generating data


class KMeansClustering:
    def __init__(self, X, num_clusters):
        self.K = num_clusters
        self.max_iterations = 100
        self.plot_figure = True
        self.num_examples = X.shape[0]
        self.num_features = X.shape[1]

    def initialize_random_centroids(self, X):
        """
        Initialize the centroids as K random samples from the dataset

        Parameters:
            X: dataset to sample the centroids from

        Returns:
            centroids: K random samples from the dataset

        """
        centroids = np.zeros((self.K, self.num_features))

        for k in range(self.K):
            centroid = X[np.random.choice(range(self.num_examples))]
            centroids[k] = centroid

        return centroids

    def create_clusters(self, X, centroids):
        """
        Creates clusters by assigning each sample to the closest centroid

        Parameters:
            X: dataset to create clusters from

        Returns:
            clusters: list of clusters where each cluster is a list of
        """

        # Will contain a list of the points that are associated with that specific cluster
        clusters = [[] for _ in range(self.K)]

        # Loop through each point and check which is the closest cluster
        for point_idx, point in enumerate(X):
            closest_centroid = np.argmin(
                np.sqrt(np.sum((point - centroids) ** 2, axis=1))
            )
            clusters[closest_centroid].append(point_idx)

        return clusters

    def calculate_new_centroids(self, clusters, X):
        """
        Calculate the new centroids as the means of the samples in each cluster

        Parameters:
            clusters: list of clusters where each cluster is a list of sample indices

        Returns:
            centroids: new centroids as the means of the samples in each cluster

        """
        centroids = np.zeros((self.K, self.num_features))
        for idx, cluster in enumerate(clusters):
            new_centroid = np.mean(X[cluster], axis=0)
            centroids[idx] = new_centroid

        return centroids

    def predict_cluster(self, clusters, X):
        """
        Predicts the cluster for each sample in X

        Parameters:
            clusters: list of clusters where each cluster is a list of sample indices
            X: dataset to predict the clusters for

        Returns:
            y_pred: predicted cluster for each sample in X
        """
        y_pred = np.zeros(self.num_examples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx

        return y_pred

    def plot_fig(self, X, y):
        """
        Plots the data and the clusters

        Parameters:
            X: dataset to plot
            y: predicted clusters for each sample in X


        """
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
        plt.show()

    def fit(self, X):
        """
        Fits the K means clustering model to the dataset X

        Parameters:
            X: dataset to fit the model to

        Returns:
            centroids: final centroids of the model
            y_pred: predicted clusters for each sample in X

        """
        centroids = self.initialize_random_centroids(X)

        for it in range(self.max_iterations):
            clusters = self.create_clusters(X, centroids)

            previous_centroids = centroids
            centroids = self.calculate_new_centroids(clusters, X)

            diff = centroids - previous_centroids

            if not diff.any():
                print("Termination criterion satisfied")
                break

        # Get label predictions
        y_pred = self.predict_cluster(clusters, X)

        if self.plot_figure:
            self.plot_fig(X, y_pred)

        return y_pred


if __name__ == "__main__":
    """
    Test the implementation of K means clustering
    and plot the results
    """

    # Generate data
    X, y = make_blobs(
        n_samples=1500,
        n_features=2,
        centers=3,
        cluster_std=0.5,
        shuffle=True,
        random_state=0,
    )

    # Create model and fit it to the data
    model = KMeansClustering(X, num_clusters=3)
    model.fit(X)

    # Plot the results
    model.plot_fig(X, y)
