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
        self.input = data_input
        self.mask = np.random.binomial(1, 1 - self.rate, size=self.input.shape)
        self.output = self.input * self.mask
        return self.output

    def backward(self, error, learning_rate):

        return error * self.mask

    def __str__(self):
        return f'DropoutLayer(rate={self.rate})'

    def params_count(self):
        return 0

    def summary(self):
        print(f'DropoutLayer: rate = {self.rate}')

    def __doc__():
        return 'Dropout layer used in neural networks to prevent overfitting by randomly setting some input units to 0'
