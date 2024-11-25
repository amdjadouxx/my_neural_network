import numpy as np
from .Layer import Layer

class DropoutLayer(Layer):
    def __init__(self, rate):
        """
        Create a dropout layer

        :param rate: float: dropout rate (0.0 to 1.0)
        """
        self.rate = rate
        self.mask = None

    def forward(self, data_input) -> np.ndarray:
        """
        Forward pass of the dropout layer

        :param data_input: np.ndarray: input data
        :return: np.ndarray: output data
        """
        self.input = data_input
        self.mask = np.random.binomial(1, 1 - self.rate, size=self.input.shape)
        self.output = self.input * self.mask
        return self.output

    def backward(self, error, learning_rate):
        """
        Backward pass of the dropout layer

        :param error: np.ndarray: error from the next layer
        :param learning_rate: float: learning rate
        
        :return: np.ndarray: error to pass to the previous layer
        """
        return error * self.mask

    def params_count(self):
        """
        Get the number of parameters in the dropout layer
        """
        return 0

    def summary(self):
        """
        Print a summary of the dropout layer
        """
        print(f'DropoutLayer: rate = {self.rate}')

    def __doc__():
        """
        Get the documentation of the dropout layer
        """
        return 'Dropout layer used in neural networks to prevent overfitting by randomly setting some input units to 0'

    def get_load_line(self):
        """
        Get the line of code to load the dropout layer
        """
        return f'DropoutLayer(rate={self.rate})'
    
DropoutLayer.__name__ = "Dropout Layer"