from scipy.sparse import hstack, csr_matrix, issparse
import numpy as np

class KNN:
    def __init__(self, distance="euclidean", n_neighbors=3, p=2):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        self.distance = distance
        self.p = p  # For Minkowski distance

    def euclidean_distance(self, X, Y):
        # Convert to dense if sparse
        if issparse(X):
            X = X.toarray()
        if issparse(Y):
            Y = Y.toarray()

        # Compute Euclidean distance
        dists = np.sqrt(((X[:, np.newaxis] - Y[np.newaxis, :]) ** 2).sum(axis=2))
        return dists

    def manhattan_distance(self, X, Y):
        # Convert to dense if sparse
        if issparse(X):
            X = X.toarray()
        if issparse(Y):
            Y = Y.toarray()

        # Compute Manhattan distance
        dists = np.abs(X[:, np.newaxis] - Y[np.newaxis, :]).sum(axis=2)
        return dists

    def minkowski_distance(self, X, Y):
        # Convert to dense if sparse
        if issparse(X):
            X = X.toarray()
        if issparse(Y):
            Y = Y.toarray()

        # Compute Minkowski distance
        dists = (np.abs(X[:, np.newaxis] - Y[np.newaxis, :]) ** self.p).sum(axis=2) ** (1 / self.p)
        return dists
    
    def get_distance(self, X, Y):
        if self.distance == "euclidean":
            return self.euclidean_distance(X, Y)
        elif self.distance == "manhattan":
            return self.manhattan_distance(X, Y)
        elif self.distance == "minkowski":
            return self.minkowski_distance(X, Y)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance}")

    def get_neighbors(self, test_points):
        # Compute distances between test points and all training points
        dists = self.get_distance(test_points, self.X_train)

        # Sort and get the k nearest neighbors
        neighbors = []
        for i in range(dists.shape[0]):
            sorted_indices = np.argsort(dists[i])
            nearest_indices = sorted_indices[:self.n_neighbors]
            neighbors.append([(dists[i, idx], self.y_train.iloc[idx]) for idx in nearest_indices])
        return neighbors

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self

    def predict(self, X_test):
        if isinstance(X_test, np.ndarray) == False:
            X_test = X_test.toarray()
        
        # Compute distances for all test points
        dists = self.get_distance(X_test, self.X_train)

        # Initialize array for predictions
        predictions = np.zeros(X_test.shape[0])

        # Calculate predictions based on k-nearest neighbors
        for i in range(X_test.shape[0]):
            sorted_indices = np.argsort(dists[i])
            nearest_indices = sorted_indices[:self.n_neighbors]
            nearest_labels = self.y_train.iloc[nearest_indices].values
            predictions[i] = np.mean(nearest_labels)
        
        return predictions
    
        
    def set_params(self, **params):
        # Update parameters based on the input dictionary
        for key, value in params.items():
            setattr(self, key, value)
        return self