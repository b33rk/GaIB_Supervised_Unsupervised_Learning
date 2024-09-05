from scipy.sparse import hstack, csr_matrix, issparse
import numpy as np
import math

class XGBoostRegression : 
    def __init__(self, base_prediction = 0.5, lamda = 0, min_child_weight = 1, gamma = 0, max_depth = 5, min_samples_split = 2, max_tree = 5, lr = 0.3, subsample = 1, random_seed = 0, verbose=False) :
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []
        self.lamda = lamda
        self.max_tree = max_tree
        self.lr = lr
        self.subsample = subsample
        self.rng = np.random.default_rng(seed=random_seed)
        self.base_prediction = base_prediction
        self.verbose = verbose
        self.min_child_weight= min_child_weight
        self.gamma = gamma
    
    def fit(self, X, y) : 
        self.n_features = X.shape[1]
        if issparse(X):
            self.X = X.toarray()        
        curr_prediction = self.base_prediction * np.ones(shape=y.shape)

        for i in range(self.max_tree):
            gradients = self.get_gradient(curr_prediction)
            hessian = self.get_hessian()
            sample_idx = None if sample_idx == 1 else self.rng.choice(len(y), 
                                 size=math.floor(self.subsample*len(y)), 
                                 replace=False)
            data_tree = self.X[sample_idx,:]
            tree = TreeBoost(data_tree, gradients, hessian, self.max_depth, self.min_samples_split, self.lr, self.lamda, self.gamma, self.min_child_weight)
            curr_prediction += self.lr * tree.predict(X)
            self.trees.append(tree)
            if self.verbose: 
                print(f'Tree [{i}] train loss = {self.get_loss(y, curr_prediction)}')
    
    def get_gradient(self, prediction):
        return prediction - self.y
    
    def get_hessian(self):
        return np.ones(len(self.y))
    
    def get_loss(self, prediction):
        return np.mean((self.y - prediction)**2)
    
    def predict(self, X):
        return (self.base_prediction + self.lr 
                * np.sum([tree.predict(X) for tree in self.trees], axis=0))
    
class TreeBoost:
    def __init__(self, data, g, h, max_depth, min_samples_split, lr, lamda, gamma, min_child_weight):
        self.data = data
        self.g, self.h = g, h
        self.lamda = lamda
        self.gain = g.sum() / (h.sum() + self.lamda)
        self.best_gain = 0
        self.row, self.n_features = data.shape
        self.max_depth = max_depth
        self.lr = lr
        self.min_samples_split = min_samples_split
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        if self.max_depth > 0:
            if self.row >= min_samples_split:
                self.build_tree()
    
    def build_tree(self):
        for feature_idx in range(self.n_features) :
            curr_data = self.data.values[:, feature_idx]
            sort_idx = np.argsort(curr_data)
            sort_g, sort_h, sort_data = self.g[sort_idx], self.h[sort_idx], curr_data[sort_idx]
            sum_g, sum_h = self.g.sum(), self.h.sum()
            sum_g_right, sum_h_right = sum_g, sum_h
            sum_g_left, sum_h_left = 0., 0.

            for i in range(0, self.rows - 1):
                g_i, h_i, data_i, data_i_next = sort_g[i], sort_h[i], sort_data[i], sort_data[i + 1]
                sum_g_left += g_i
                sum_g_right -= g_i
                sum_h_left += h_i
                sum_h_right -= h_i
                if (sum_h_left < self.min_child_weight or data_i == data_i_next):
                    continue
                if (sum_h_right < self.min_child_weight): 
                    break
                gain = 0.5 * ((sum_g_left**2 / (sum_h_left + self.lamda))
                        + (sum_g_right**2 / (sum_h_right + self.lamda))
                        - (sum_g**2 / (sum_h + self.lamda))
                        ) - self.gamma/2
                if gain > self.best_gain: 
                    self.split_feature_idx = feature_idx
                    self.best_gain = gain
                    self.threshold = (data_i + data_i_next) / 2
        left_idx = np.nonzero(self.data <= self.threshold)[0]
        right_idx = np.nonzero(self.data > self.threshold)[0]
        left, right = self.data[left_idx, self.split_feature_idx], self.data[right_idx, self.split_feature_idx]
        g_left, h_left = self.g[left_idx], self.h[left_idx]
        g_right, h_right = self.g[right_idx], self.h[right_idx]
        self.left = TreeBoost(left, g_left, h_left, self.max_depth, self.min_samples_split, self.lr, self.lamda, self.gamma, self.min_child_weight)
        self.right = TreeBoost(right, g_right, h_right, self.max_depth, self.min_samples_split, self.lr, self.lamda, self.gamma, self.min_child_weight)

    def is_leaf(self):
        return self.best_gain == 0      
        
    def get_similiarity(self, residual):
        return np.square(np.sum(residual)) / (len(residual) + self.lamda)
        
    def split_data(self, data, feature_idx, thresh) : 
        return  data[data[:, feature_idx] <= thresh], data[data[:, feature_idx] > thresh]
    
    def predict_one(self, row):
        if self.is_leaf: 
            return self.gain
        child = self.left if (row[self.split_feature_idx] <= self.threshold) else self.right
        return child.predict_one(row)
    
    def predict(self, X):
        return np.array([self.predict_one(row) for i, row in X.iterrows()])