import numpy as np

class Reshape:
    def __init__(self, output_shape):
        self.input_shape = None
        self.output_shape = output_shape

    def forward(self, input):
        if self.input_shape is None:
            self.input_shape = input.shape[1:]
        self.batch_size = input.shape[0]
        return np.reshape(input, (self.batch_size, *self.output_shape))

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, (self.batch_size, *self.input_shape))