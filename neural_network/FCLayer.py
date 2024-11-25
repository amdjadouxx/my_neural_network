import numpy as np
from .Layer import Layer

class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        """
        Fully connected layer

        :param input_size: int: number of inputs
        :param output_size: int: number of outputs
        """
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.random.rand(1, output_size) - 0.5

    def forward(self, data_input) -> np.ndarray:
        """
        Forward pass

        :param data_input: np.ndarray: input data
        :return: np.ndarray: output data
        """
        self.input = data_input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward(self, error, learning_rate) -> np.ndarray:
        """
        Backward pass

        :param error: np.ndarray: error data
        :param learning_rate: float: learning rate
        :return: np.ndarray: output error
        """
        weights_error = np.dot(self.input.T, error)

        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * error

        return np.dot(error, self.weights.T)

    def params_count(self) -> int:
        """
        Number of parameters
        """
        return self.weights.size + self.biases.size

    def summary(self):
        """
        Summary
        """
        print(f'Fully connected layer')
        print(f'Paramètres: {self.params_count()}')

    def __doc__(self) -> str:
        """
        Documentation
        """
        return 'Fully connected layer used in neural networks to weight inputs and add biases to them'

    def get_load_line(self) -> str:
        """
        Get the line of code to load the layer
        """
        return f'FCLayer({self.weights.shape[0]}, {self.weights.shape[1]})'
    
FCLayer.__name__ = "Fully Connected Layer"