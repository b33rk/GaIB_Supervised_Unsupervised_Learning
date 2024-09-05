from scipy.sparse import hstack, csr_matrix, issparse
import numpy as np

class SVRegression:
    def __init__(self, gamma='scale', C=1.0, epsilon=0.1, kernel='linear', degree=3, learning_rate=0.01, max_iter=1000):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)

    def polynomial_kernel(self, X1, X2):
        return (1 + np.dot(X1, X2.T)) ** self.degree

    def rbf_kernel(self, X1, X2):
        sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * sq_dists)

    def kernel_function(self, X1, X2):
        if self.kernel == 'linear':
            return self.linear_kernel(X1, X2)
        elif self.kernel == 'poly':
            return self.polynomial_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return self.rbf_kernel(X1, X2)
        else:
            raise ValueError("Unknown kernel")

    def fit(self, X, y):
        if issparse(X):
            X = X.toarray()

        n_samples, n_features = X.shape

        # Set gamma based on the input or the number of features
        if isinstance(self.gamma, str):
            if self.gamma == 'scale':
                self.gamma = 1 / (n_features * X.var())
            elif self.gamma == 'auto':
                self.gamma = 1 / n_features
            else:
                raise ValueError("Unknown gamma value")

        # Initialize alpha, b
        self.alpha = np.zeros(n_samples)
        self.b = 0

        # Calculate the Kernel matrix
        K = self.kernel_function(X, X)

        # Gradient Descent
        for iteration in range(self.max_iter):
            for i in range(n_samples):
                prediction = np.dot(K[i], self.alpha) + self.b
                error = prediction - y.iloc[i]

                # Compute gradients
                if np.abs(error) > self.epsilon:
                    self.alpha[i] -= self.learning_rate * (error + self.C * np.sign(self.alpha[i]))
                    self.b -= self.learning_rate * error

        support_vectors_idx = np.where(self.alpha != 0)[0]
        self.support_vectors = X[support_vectors_idx]
        self.support_vector_labels = y.iloc[support_vectors_idx]
        self.support_vector_alphas = self.alpha[support_vectors_idx]

    def predict_single(self, X):
        kernel = self.kernel_function(self.support_vectors, X)
        return np.dot(kernel.T, self.support_vector_alphas) + self.b

    def predict(self, X):
        if issparse(X):
            X = X.toarray()
        return np.array([self.predict_single(x) for x in X])
    
    def set_params(self, **params):
        # Update parameters based on the input dictionary
        for key, value in params.items():
            setattr(self, key, value)
        return self
