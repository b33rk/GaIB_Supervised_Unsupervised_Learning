from scipy.sparse import hstack, csr_matrix, issparse
import numpy as np

class PolynomialRegression():
    def __init__(self, degree=2, lr=0.01, reg_term=None, i=100, alpha=0.1, verbose=0, optimization="gradient_descent"):
        # initialize intercept and coeffecient parameter
        self.intercept = 1
        self.coeffecients = None

        self.degree = degree
        self.lr = lr
        self.i = i
        self.reg_term = reg_term
        self.alpha = alpha
        self.verbose = verbose
        self.optimization = optimization
    
    def _expand_features(self, X):
        X_poly = csr_matrix(np.ones((X.shape[0], 1)))

        # Expand features
        for d in range(1, self.degree + 1):
            X_power = X.power(d)  # Compute the power of each feature in X
            X_poly = hstack([X_poly, X_power])
        
        return X_poly
    
    def loss_function(self, pred, Y):
        loss_ori = np.sum(np.square(Y - pred))/Y.size

        if (self.reg_term == 'l1'):
            return loss_ori + self.alpha * np.sum(np.abs(self.coeffecients))
        elif (self.reg_term == 'l2'):
            return loss_ori + self.alpha * np.sum(np.square(self.coeffecients))
        
        return loss_ori
    
    def derivative_coeffecient(self, X, y, Y_pred):
        m = y.size
        error = y - Y_pred
        derivative_ori = (-2/m) * X.T.dot(error)
        
        return self._apply_reg_term(derivative_ori)
    
    def _apply_reg_term(self, derivative):
        if (self.reg_term == 'l1'):
            coeffecient_dense = self.coeffecients.toarray()
            l1_reg = self.alpha * np.sign(coeffecient_dense).flatten()
            return derivative + self.alpha * csr_matrix(l1_reg).T
        elif (self.reg_term == 'l2'):
            return derivative + self.alpha * 2 * self.coeffecients
        return derivative
    
    def hessian_matrix(self, X_poly):
        m = X_poly.shape[0]
        H = (2/m) * (X_poly.T.dot(X_poly))
        
        if self.reg_term == 'l2': 
            H += 2 * self.alpha * csr_matrix(np.eye(X_poly.shape[1]))
        return H
    
    def fit(self, X, y):
        if not isinstance(X, csr_matrix):
            X = csr_matrix(X)
        X_poly = self._expand_features(X)
        y = y.values.reshape(-1,1)

        m, n = X_poly.shape
        self.coeffecients = csr_matrix(np.zeros((n, 1)))
        
        for iteration in range(self.i):
            Y_pred = self.predict(X)  # The current predicted value of Y
            D_c = self.derivative_coeffecient(X_poly, y, Y_pred)  # Gradient
            D_m = (-2/m) * np.sum(y - Y_pred)  # Derivative wrt intercept
            self.intercept -= self.lr * np.clip(D_m, -1e5, 1e5)  # Clip gradients
            
            if self.optimization == 'gradient_descent':
                # Gradient Descent Update
                self.coeffecients -= csr_matrix(self.lr * np.clip(D_c, -1e5, 1e5))  # Clip gradients
            
            elif self.optimization == 'newtons_method':
                # Newton's Method Update
                H = self.hessian_matrix(X_poly)  # Hessian matrix
                try:
                    H_inv = np.linalg.inv(H.toarray())
                    coef_update = csr_matrix(H_inv.dot(D_c))
                    self.coeffecients -= coef_update
                except np.linalg.LinAlgError:
                    print(f"Iteration {iteration}: Hessian is singular, switching to gradient descent for this step.")
                    self.coeffecients -= csr_matrix(self.lr * np.clip(D_c, -1e5, 1e5))

            if (iteration % 1000 == 0 and self.verbose):
                print(f"Iteration: {iteration}, Loss: {self.loss_function(Y_pred, y)}")

    def predict(self, X):
        if not isinstance(X, csr_matrix):
            X = csr_matrix(X)
        X_poly = self._expand_features(X)
        return X_poly.dot(self.coeffecients).A + self.intercept
    
    def set_params(self, **params):
        # Update parameters based on the input dictionary
        for key, value in params.items():
            setattr(self, key, value)
        return self