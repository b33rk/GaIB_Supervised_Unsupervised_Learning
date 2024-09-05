import numpy as np

class MaxPooling2D:
    def __init__(self, pool_size=(2, 2), stride=(2, 2)):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        self.batch_size, self.input_height, self.input_width, self.input_channels = input.shape
        self.input = input

        self.output_height = (self.input_height - self.pool_size[0]) // self.stride[0] + 1
        self.output_width = (self.input_width - self.pool_size[1]) // self.stride[1] + 1

        # Initialize output tensor
        output = np.zeros((self.batch_size, self.output_height, self.output_width, self.input_channels))

        for i in range(self.output_height):
            for j in range(self.output_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.pool_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.pool_size[1]

                region = self.input[:, h_start:h_end, w_start:w_end, :]
                output[:, i, j, :] = np.max(region, axis=(1, 2))
        
        return output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.zeros_like(self.input)

        for i in range(self.output_height):
            for j in range(self.output_width):
                h_start = i * self.stride[0]
                h_end = h_start + self.pool_size[0]
                w_start = j * self.stride[1]
                w_end = w_start + self.pool_size[1]

                region = self.input[:, h_start:h_end, w_start:w_end, :]
                max_mask = (region == np.max(region, axis=(1, 2), keepdims=True))
                
                input_gradient[:, h_start:h_end, w_start:w_end, :] += max_mask * output_gradient[:, i, j, :][:, None, None, :]
        
        return input_gradient
