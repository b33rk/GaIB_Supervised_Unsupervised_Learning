import numpy as np

class LossFunction:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mse_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size
    
    @staticmethod
    def log_loss(y_true, y_pred, epsilon=1e-15):
        # Clip the predictions to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def log_loss_derivative(y_true, y_pred, epsilon=1e-15):
        # Clip the predictions to avoid division by 0
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)