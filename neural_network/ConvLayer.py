from .Layer import Layer
from scipy import signal
import numpy as np

class ConvLayer(Layer):

    def __init__(self, input_shape, kernel_shape, layer_depth):
        """
        Initialize the Convolutional Layer

        :param input_shape: tuple of 3 integers (height, width, depth)
        :param kernel_shape: tuple of 2 integers (height, width)
        :param layer_depth: integer
        """
        self.input_shape = input_shape
        self.input_depth = input_shape[2]
        self.kernel_shape = kernel_shape
        self.layer_depth = layer_depth
        self.output_shape = (input_shape[0]-kernel_shape[0]+1, input_shape[1]-kernel_shape[1]+1, layer_depth)
        self.weights = np.random.rand(kernel_shape[0], kernel_shape[1], self.input_depth, layer_depth) - 0.5
        self.bias = np.random.rand(layer_depth) - 0.5

    def forward(self, input):
        """
        Forward pass of the Convolutional Layer

        :param input: 3D numpy array
        """
        self.input = input
        self.output = np.zeros(self.output_shape)

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                self.output[:,:,k] += signal.correlate2d(self.input[:,:,d], self.weights[:,:,d,k], 'valid') + self.bias[k]

        return self.output

    def backward(self, output_error, learning_rate):
        """
        Backward pass of the Convolutional Layer

        :param output_error: 3D numpy array
        :param learning_rate: float
        """
        in_error = np.zeros(self.input_shape)
        dWeights = np.zeros((self.kernel_shape[0], self.kernel_shape[1], self.input_depth, self.layer_depth))
        dBias = np.zeros(self.layer_depth)

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                in_error[:,:,d] += signal.convolve2d(output_error[:,:,k], self.weights[:,:,d,k], 'full')
                dWeights[:,:,d,k] = signal.correlate2d(self.input[:,:,d], output_error[:,:,k], 'valid')
            dBias[k] = self.layer_depth * np.sum(output_error[:,:,k])

        self.weights -= learning_rate*dWeights
        self.bias -= learning_rate*dBias
        return in_error
    
    def params_count(self):
        """
        Return the number of parameters in the Convolutional Layer
        """
        return np.prod(self.weights.shape) + np.prod(self.bias.shape)
    
    def summary(self):
        """
        Print the summary of the Convolutional Layer
        """
        print("Convolutional Layer")
        print("Input Shape: ", self.input_shape)
        print("Kernel Shape: ", self.kernel_shape)
        print("Layer Depth: ", self.layer_depth)
        print("Output Shape: ", self.output_shape)
        print("Weights Shape: ", self.weights.shape)
        print("Bias Shape: ", self.bias.shape)
    
    def __doc__(self):
        """
        Return the description of the layer
        """
        return "Convolutional Layer used in neural networks to detect features in images"
    
    def get_load_line(self):
        """
        Return the line of code to load the layer
        """
        return f'ConvLayer({self.input_shape}, {self.kernel_shape}, {self.layer_depth})'
    
ConvLayer.__name__ = 'ConvLayer'