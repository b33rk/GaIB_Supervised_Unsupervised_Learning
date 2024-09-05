import numpy as np
        
# Layer Class
class DenseLayer:
    def __init__(self, output_size, init=None, reg_type=None, reg_param=0.0):
        self.reg_type = reg_type
        self.reg_param = reg_param
        self.weights = None
        self.biases = np.zeros((1, output_size))
        self.init = init
        self.output_size = output_size
    
    def _initialize_weights(self, input_size):
        # Weight Initialization
        if self.init is None or self.init == "random":
            self.weights = np.random.randn(input_size, self.output_size) - 0.5
        elif self.init == "Xavier":
            self.weights = np.random.randn(input_size, self.output_size) * np.sqrt(1. / input_size)
        elif self.init == "He":
            self.weights = np.random.randn(input_size, self.output_size) * np.sqrt(2. / input_size)
        elif self.init == "Zero":
            self.weights = np.zeros((input_size, self.output_size))
        elif self.init == "One":
            self.weights = np.ones((input_size, self.output_size))
        else:
            raise ValueError(f"Unknown initialization method: {self.init}")
    
    def forward(self, input):
        if len(input.shape) > 2:
            input = np.reshape(input, (input.shape[0], input.shape[1]))
        if self.weights is None:
            self.m, self.input_size = input.shape
            self._initialize_weights(self.input_size)
        self.input = input
        return np.dot(self.input, self.weights) + self.biases
    
    def backward(self, output_gradient, learning_rate):
        input_gradient = np.dot(output_gradient, self.weights.T)
        weight_gradient = np.dot(self.input.T, output_gradient) / self.m
        
        # Apply regularization
        if self.reg_type == 'l1':
            weight_gradient += self.reg_param * np.sign(self.weights)
        elif self.reg_type == 'l2':
            weight_gradient += self.reg_param * self.weights
        
        self.weights -= learning_rate * weight_gradient
        self.biases -=  learning_rate * np.sum(output_gradient, axis=0, keepdims=True) / self.m
        
        return input_gradient