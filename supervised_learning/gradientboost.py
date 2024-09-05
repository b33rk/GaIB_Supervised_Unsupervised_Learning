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
        self.gain = 0
        self.level = -1
    
    def set_leaf(self, result, level) :
        self.is_leaf = True
        self.result = result
        self.level = level
    
    def set_branch(self, best_feature, best_feature_thresh, best_gain, left, right, level) :
        self.feature_id = best_feature
        self.feature_thresh = best_feature_thresh
        self.gain = best_gain
        self.left = left
        self.right = right
        self.level = level
    
    def traverse(self, inp):
        if self.is_leaf:
            return self.result, True
        if inp[self.feature_id] <= self.feature_thresh:
            return self.left.traverse(inp)
        else:
            return self.right.traverse(inp)

class GradientBoostRegression : 
    def __init__(self, lamda = 0, max_depth = 5, min_samples_leaf = 1, min_samples_split = 2, max_tree = 5, lr = 0.3) :
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = []
        self.lamda = lamda
        self.max_tree = max_tree
        self.lr = lr
    
    def fit(self, X, y) : 
        self.prediction = [0.5 for x in range(len(y))]
        self.n_features = X.shape[1]
        if issparse(X):
            self.x = X.toarray()
        else:
            self.x = np.array(X)
        self.y = np.array(y)
        self.classes = set(y)
        self.residual = self.get_residual()
        self.data = np.concatenate((self.x, self.residual.reshape(-1, 1)), axis=1)
        for i in range(self.max_tree):
            root = self.buildTree(0, self.data)
            self.root.append(root)
            self.update_prediction()
            self.data[:, -1] = self.get_residual()
            print(i)
    
    def update_prediction(self):
        for i in range(len(self.y)): 
            for tree in self.root:
                res, _ = tree.traverse(self.x[i])
                self.prediction[i] += self.lr * self.get_output(res)
            
    def get_residual(self):
        return self.y - self.prediction
    
    def buildTree(self, level, data):
        n = Node()
        if level == self.max_depth or len(set(data[:, -1]))==1 or len(data) < self.min_samples_split: 
            n.set_leaf(data[:, -1], level)
        else : 
            best_gain = float("inf")
            left_partition = None
            right_partition = None
            best_feature = None
            best_feature_thresh = None
            root_similiarity = self.get_similiarity(data)
            for feature_idx in range(self.n_features) :
                sorted_data = sorted(set(data[:, feature_idx]))
                for i in range(len(sorted_data) - 1):
                    thresh = (sorted_data[i] + sorted_data[i + 1]) / 2
                    left, right = self.split_data(data, feature_idx, thresh)
                    if len(left) >= self.min_samples_leaf and len(right) >= self.min_samples_leaf:
                        left_similiarity = self.get_similiarity(left)
                        right_similiarity = self.get_similiarity(right)
                        gain = left_similiarity + right_similiarity - root_similiarity
                        if gain < best_gain:
                            best_gain = gain
                            best_feature_thresh = thresh
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
                    best_gain,
                    self.buildTree(level + 1, left_partition),
                    self.buildTree(level + 1, right_partition),
                    level
                )
        return n
    
    def get_output(self, leaf):
        return np.sum(leaf) / (len(leaf) + self.lamda)
    
    def split_data(self, data, feature_idx, thresh) : 
        return  data[data[:, feature_idx] <= thresh], data[data[:, feature_idx] > thresh]
    
    def get_similiarity(self, residual):
        return np.square(np.sum(residual)) / (len(residual) + self.lamda)
    
    def predict(self, inputs) :
        if issparse(inputs):
            inputs = inputs.toarray()
        if len(self.root) == 0: 
            raise Exception("Tree Not Fit Yet!")
        
        predictions = np.full(inputs.shape[0], 0.5)
        for i in range(inputs.shape[0]):
            for tree in self.root:
                res, _ = tree.traverse(inputs[i])
                predictions[i] += self.lr * self.get_output(res)
        return predictions