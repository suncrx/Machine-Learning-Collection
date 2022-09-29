"""
Implementation of K-nearest neighbor (KNN) from scratch
where you can either use 2-loops (inefficient), 1-loop (better)
or a heavily vectorized zero-loop implementation.

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
*    2020-04-24 Initial coding
*    2022-09-29 Ensured code works as intended and added docstrings
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class KNearestNeighbor:
    def __init__(self, k):
        self.k = k
        self.eps = 1e-8

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test, num_loops=0):
        """
        Predict labels for test data using KNN

        Parameters:
            X_test: numpy array of shape (num_test, D)
                    Test data consisting of num_test samples each of dimension D.
            num_loops: determines which implementation to use to compute distances

        Returns:
            y: numpy array of shape (num_test,)
                Predicted labels for the test data, where y[i] is the predicted label
                for the test point X[i].

        """
        if num_loops == 0:
            distances = self.compute_distance_vectorized(X_test)

        elif num_loops == 1:
            distances = self.compute_distance_one_loop(X_test)

        else:
            distances = self.compute_distance_two_loops(X_test)

        return self.predict_labels(distances)

    def compute_distance_two_loops(self, X_test):
        """
        Naive implementation of KNN using two loops

        Parameters:
            X_test: numpy array of shape (num_test, D)

        Returns:
            distances: numpy array of shape (num_test, num_train)
        """

        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                # (Taking sqrt is not necessary: min distance won't change since sqrt is monotone)
                distances[i, j] = np.sqrt(
                    self.eps + np.sum((X_test[i, :] - self.X_train[j, :]) ** 2)
                )

        return distances

    def compute_distance_one_loop(self, X_test):
        """
        Naive implementation of KNN using one loop

        Parameters:
            X_test: numpy array of shape (num_test, D)

        Returns:
            distances: numpy array of shape (num_test, num_train)
        """
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))

        for i in range(num_test):
            # (Taking sqrt is not necessary: min distance won't change since sqrt is monotone)
            distances[i, :] = np.sqrt(
                self.eps + np.sum((self.X_train - X_test[i, :]) ** 2, axis=1)
            )

        return distances

    def compute_distance_vectorized(self, X_test):
        """
        Can be tricky to understand this, we utilize heavy
        vecotorization as well as numpy broadcasting.
        Idea: if we have two vectors a, b (two examples)
        and for vectors we can compute (a-b)^2 = a^2 - 2a (dot) b + b^2
        expanding on this and doing so for every vector lends to the
        heavy vectorized formula for all examples at the same time.

        Parameters:
            X_test: numpy array of shape (num_test, D)

        Returns:
            distances: numpy array of shape (num_test, num_train)
        """
        X_test_squared = np.sum(X_test**2, axis=1, keepdims=True)
        X_train_squared = np.sum(self.X_train**2, axis=1, keepdims=True)
        two_X_test_X_train = np.dot(X_test, self.X_train.T)

        # (Taking sqrt is not necessary: min distance won't change since sqrt is monotone)
        return np.sqrt(
            self.eps + X_test_squared - 2 * two_X_test_X_train + X_train_squared.T
        )

    def predict_labels(self, distances):
        """
        Given distances between test data and training data, predict labels

        Parameters:
            distances: numpy array of shape (num_test, num_train)

        Returns:
            y: numpy array of shape (num_test,)
        """
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):
            y_indices = np.argsort(distances[i, :])
            k_closest_classes = self.y_train[y_indices[: self.k]].astype(int)
            y_pred[i] = np.argmax(np.bincount(k_closest_classes))

        return y_pred


if __name__ == "__main__":
    """
    Testing the KNN implementation
    """

    # Load the dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train the model
    knn = KNearestNeighbor(k=3)
    knn.train(X_train, y_train)

    # Predict the labels
    y_pred = knn.predict(X_test, num_loops=0)

    # Compute the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")