import numpy as np

class Activation:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return self.activation_prime(self.input) * output_gradient

# Activation Functions
class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

class ReLu(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return x > 0

        super().__init__(relu, relu_prime)

class Linear(Activation):
    def __init__(self):
        def linear(x):
            return x

        def linear_prime(x):
            return np.ones_like(x)

        super().__init__(linear, linear_prime)

class Softmax:
    def __init__(self, softmax_logloss=False):
        self.softmax_logloss = softmax_logloss
    
    def forward(self, input):
        exps = np.exp(input - np.max(input, axis=1, keepdims=True))
        output = exps / np.sum(exps, axis=1, keepdims=True)
        return output
    
    def backward(self, output_gradient, learning_rate):
        if self.softmax_logloss:
            return output_gradient
        # print(output_gradient.shape)
        n = self.output.shape[0]
        grad_input = np.empty_like(output_gradient)
        
        for i in range(n):
            y = self.output[i].reshape(-1, 1)
            jacobian_matrix = np.diagflat(y) - np.dot(y, y.T)
            grad_input[i] = np.dot(jacobian_matrix, output_gradient[i])

        return output_gradient