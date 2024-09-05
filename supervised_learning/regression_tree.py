from scipy.sparse import hstack, csr_matrix, issparse
import numpy as np

class Node : 
    def __init__(self) :
        self.left = None
        self.right = None
        self.feature_id = None
        self.feature_thresh = None
        self.is_leaf = False
        self.result = None
        self.metric = 0
        self.level = -1
    
    def set_leaf(self, result, level) :
        self.is_leaf = True
        self.result = result
        self.level = level
    
    def set_branch(self, best_feature, best_feature_thresh, best_metric, left, right, level) :
        self.feature_id = best_feature
        self.feature_thresh = best_feature_thresh
        self.metric = best_metric
        self.left = left
        self.right = right
        self.level = level
    
    def traverse(self, inp):
        if self.is_leaf : 
            return self.result, True
        if inp[self.feature_id] <= self.feature_thresh : 
            return self.left, False
        else : 
            return self.right, False

class RegressionTree : 
    def __init__(self, max_depth = 5, min_samples_leaf = 1, min_samples_split = 2) :
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None
    
    def fit(self, X, y) : 
        self.n_features = X.shape[1]
        if issparse(X):
            self.x = X.toarray()
        else:
            self.x = np.array(X)
        self.y = np.array(y)
        self.classes = set(y)
        self.data = np.concatenate((self.x, self.y.reshape(-1, 1)), axis=1)
        self.root = self.buildTree(0, self.data)
    
    def get_metric(self, y) :
        total = len(y)
        return np.sum((y-y.mean())**2)/total
    
    def split_data(self, data, feature_idx, thresh) : 
        return  data[data[:, feature_idx] <= thresh], data[data[:, feature_idx] > thresh]
    
    def buildTree(self, level, data) :
        n = Node()
        if level == self.max_depth or len(set(data[:, -1]))==1 or len(data) < self.min_samples_split: 
            n.set_leaf(np.mean(data[:, -1]), level)
        else : 
            best_metric = float("inf")
            left_partition = None
            right_partition = None
            best_feature = None
            best_feature_thresh = None
            for feature_idx in range(self.n_features) :
                for thresh in set(self.data[:, feature_idx]) : 
                    left, right = self.split_data(data, feature_idx, thresh)
                    if len(left) >= self.min_samples_leaf and len(right) >= self.min_samples_leaf:
                        m = len(left)*self.get_metric(left[:, -1]) + len(right)*self.get_metric(right[:, -1])
                        if m < best_metric : 
                            best_metric = m
                            left_partition = left
                            right_partition = right
                            best_feature = feature_idx
                            best_feature_thresh = thresh
            if best_feature is None:
                n.set_leaf(np.mean(data[:, -1]), level)
            else:
                n.set_branch(
                    best_feature,
                    best_feature_thresh,
                    best_metric,
                    self.buildTree(level + 1, left_partition),
                    self.buildTree(level + 1, right_partition),
                    level
                )
        return n
    
    def predict(self, inputs, log_track=False) :
        if issparse(inputs):
            inputs = inputs.toarray()
        if not self.root : 
            raise Exception("Tree Not Fit Yet!")
        results = []
        for inp in inputs : 
            res = self.root
            fin = False
            while not fin : 
                if log_track:
                    print("\n\n==== Node ====")
                    print(res)
                    if not res.is_leaf : 
                        print("Input Value here : ", inp[res.feature_id])
                        if inp[res.feature_id] < res.feature_thresh : 
                            print("Taking Left")
                        else : 
                            print("Taking Right")
                    else : 
                        print("Decision : ", res.result)
                res, fin = res.traverse(inp)
            results.append(res)
        return results
    
    def set_params(self, **params):
        # Update parameters based on the input dictionary
        for key, value in params.items():
            setattr(self, key, value)
        return self