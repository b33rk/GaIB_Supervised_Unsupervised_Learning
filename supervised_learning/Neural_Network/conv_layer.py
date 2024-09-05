import numpy as np
from scipy import signal

class Convolutional2D:
    def __init__(self, kernel_size, num_kernels, padding=0, stride=1, kernels=None):
        self.input_shape = None
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.biases = np.random.randn(num_kernels)
        self.kernels = kernels
    
    def _init_params(self, input_shape):
        self.batch_size, *self.input_shape = input_shape
        input_height, input_width, input_channels = self.input_shape[0], self.input_shape[1], self.input_shape[2]
        self.output_height = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        self.output_width = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        self.kernels_shape = (self.kernel_size, self.kernel_size, input_channels, self.num_kernels)
        if self.kernels is None:
            self.kernels = np.random.randn(self.kernel_size, self.kernel_size, input_channels, self.num_kernels) * 0.01

    def _pad_input(self, input):
        if self.padding > 0:
            padded_input = np.pad(
                input,
                ((0, 0),
                 (self.padding, self.padding),
                 (self.padding, self.padding),
                 (0, 0)),
                mode='constant'
            )
            return padded_input
        return input

    def forward(self, input):
        if self.input_shape is None:
            self._init_params(input.shape)

        self.input = self._pad_input(input)
        self.output = np.zeros((self.batch_size, self.output_height, self.output_width, self.num_kernels))
        
        for i in range(self.batch_size):
            for k in range(self.num_kernels):
                conv_result = np.sum([
                    signal.correlate2d(self.input[i, :, :, c], self.kernels[:, :, c, k], mode='valid')
                    for c in range(self.input.shape[3])
                ], axis=0)
                self.output[i, :, :, k] = conv_result[::self.stride, ::self.stride] + self.biases[k]
        
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros_like(self.kernels)
        input_gradient = np.zeros_like(self.input)

        for i in range(self.batch_size):
            for k in range(self.num_kernels):
                upsampled_gradient = np.zeros((
                    (output_gradient.shape[1] - 1) * self.stride + self.kernel_size,
                    (output_gradient.shape[2] - 1) * self.stride + self.kernel_size
                ))
                upsampled_gradient[::self.stride, ::self.stride] = output_gradient[i, :, :, k]
                
                for c in range(self.input.shape[3]):
                    kernels_gradient[:, :, c, k] += signal.correlate2d(
                        self.input[i, :, :, c], upsampled_gradient, mode='valid'
                    )
                    input_gradient[i, :, :, c] += signal.convolve2d(
                        upsampled_gradient, self.kernels[:, :, c, k], mode='full'
                    )
        
        self.kernels -= learning_rate * kernels_gradient / self.batch_size
        self.biases -= learning_rate * np.sum(output_gradient, axis=(0, 1, 2)) / self.batch_size
        
        return input_gradient
