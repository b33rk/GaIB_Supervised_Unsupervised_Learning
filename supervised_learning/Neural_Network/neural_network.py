import numpy as np
# from loss_function import LossFunction

# ANN Class
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loss_functions = {
            'mse': (LossFunction.mse, LossFunction.mse_derivative),
            'log_loss': (LossFunction.log_loss, LossFunction.log_loss_derivative)
        }
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def compile(self, loss):
        self.loss = loss
        if loss not in self.loss_functions:
            raise ValueError("Loss function not supported.")
        self.loss_function, self.loss_derivative = self.loss_functions[loss]
    
    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, output):
        for layer in reversed(self.layers):
            output = layer.backward(output, self.learning_rate)
    
    def train(self, X, Y, epochs = 10000, learning_rate = 0.1, batch_size = 50, softmax_logloss=False, isOne_hot=False):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        n_batches = int(np.ceil(X.shape[0] / batch_size))
        y = Y.copy()
        if isOne_hot:
            Y = one_hot(Y)

        for epoch in range(epochs):
            total_error = 0
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, X.shape[0])
                
                X_batch = X[start:end]
                y_batch = Y[start:end]

                # Forward pass for the entire batch
                output = self.forward(X_batch)
                
                # Calculate batch loss and gradients
                error = self.loss_function(y_batch, output)
                if softmax_logloss:
                    grad = (output - y_batch) / batch_size
                else:
                    grad = self.loss_derivative(y_batch, output)
                
                # Backward pass
                self.backward(grad)
                
                total_error += error
            total_error /= n_batches
            if (epoch % 10 == 0):
                output = self.forward(X)
                pred = get_predictions(output)
                print(f'Epoch {epoch}, Loss: {total_error}')
                if (one_hot):
                    print(f'Epoch {epoch}, accuracy: {get_accuracy(pred, y)}')
    
    def predict(self, X):
        return self.forward(X)

def print_parameters(layer, num_elements=5):
    print(f"First {num_elements} weights: {layer.weights.flatten()[:num_elements]}")
    print(f"First {num_elements} biases: {layer.biases.flatten()[:num_elements]}")

def get_predictions(A2):
    return np.argmax(A2, axis=1)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y

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