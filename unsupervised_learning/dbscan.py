import numpy as np
import pandas as pd

class DBScan:
    def __init__(self, MinPts, epsilon, p_value = 2, dist_metric = "euclidean", epoch=1000):
        self.epoch = epoch
        self.minPts = MinPts
        self.dist_metric = dist_metric
        self.epsilon = epsilon
        self.p_value = p_value
    
    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        self.labels = [0 for i in range(len(X))]
        self.data = X
        self.cluster = 0

        for point in range(len(self.labels)):
            if (self.labels[point] != 0):
                continue

            NeighborPts = self.find_neighbor(point)

            if len(NeighborPts) < self.minPts:
                self.labels[point] = -1
            else:
                self.cluster += 1
                self.expand_cluster(NeighborPts, point)
    
    def predict(self):
        return self.labels
    
    def fit_predict(self, X):
        self.fit(X)
        return self.predict()

    def find_neighbor(self, point):
        neighbors = []
        for p in range(len(self.data)):
            if self.calculate_distance(self.data[point], self.data[p]) < self.epsilon:
                neighbors.append(p)
        
        return neighbors
    
    def calculate_distance(self, point1, point2):
        if self.dist_metric == 'euclidean':
            return np.linalg.norm(point1 - point2)
        elif self.dist_metric == 'manhattan':
            return np.sum(np.abs(point1 - point2))
        elif self.dist_metric == 'minkowski':
            return np.sum(np.abs(point1 - point2) ** self.p_value) ** (1 / self.p_value)
        else:
            raise ValueError("Unsupported distance metric. Use 'euclidean', 'manhattan', or 'minkowski'.")
    
    def expand_cluster(self, neighbors, point):
        self.labels[point] = self.cluster
        
        i = 0
        while i < len(neighbors):
            curr_point = neighbors[i]

            if self.labels[curr_point] == -1:
                self.labels[curr_point] = self.cluster
            elif self.labels[curr_point] == 0:
                self.labels[curr_point] = self.cluster

                curr_point_neighbors = self.find_neighbor(curr_point)
                if len(curr_point_neighbors) > self.minPts:
                    neighbors += curr_point_neighbors
                else:
                    self.labels[point] = -1
            
            i += 1