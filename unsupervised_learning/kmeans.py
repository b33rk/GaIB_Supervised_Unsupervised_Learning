import numpy as np
import pandas as pd

class Kmeans:
    def __init__(self, n_clusters, init=None, max_iters=10000, random_state=32):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.inertia_ = None
        self.init = init
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)  # Initialize the RNG

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        # Initialize centroids using K-means++
        self.centroids = self._initialize_centroids(X)
        
        for _ in range(self.max_iters):
            # Assign each data point to the nearest centroid
            labels = self._assign_labels(X)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check for convergence
            if np.all(self.centroids == new_centroids):
                break
                
            self.centroids = new_centroids
        
        # Calculate final inertia
        self.inertia_ = self._calculate_inertia(X, labels)
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return self._assign_labels(X)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def _assign_labels(self, X):
        # Compute distances from each data point to centroids
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

        # Assign labels based on the nearest centroid
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        # Calculate new centroids as the mean of the points assigned to each cluster
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centroids

    def _initialize_centroids(self, X):
        if self.init == "k-means++":
            # Initialize the first centroid randomly using RNG
            centroids = [X[self.rng.choice(X.shape[0])]]
            
            for _ in range(1, self.n_clusters):
                # Calculate the distance of each point to its nearest centroid
                distances = np.min(np.linalg.norm(X[:, np.newaxis] - np.array(centroids), axis=2), axis=1)
                
                # Choose the next centroid with a probability proportional to the distance squared
                probs = distances**2 / np.sum(distances**2)
                next_centroid = X[self.rng.choice(X.shape[0], p=probs)]
                centroids.append(next_centroid)
            return np.array(centroids)
        else:
            return X[self.rng.choice(X.shape[0], self.n_clusters, replace=False)]
        
    
    def _calculate_inertia(self, X, labels):
        # Compute the sum of squared distances from each point to its assigned centroid
        inertia = 0.0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            centroid = self.centroids[i]
            inertia += np.sum((cluster_points - centroid) ** 2)
        return inertia
