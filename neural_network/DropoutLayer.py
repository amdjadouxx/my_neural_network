import numpy as np
from .Layer import Layer

class DropoutLayer(Layer):
    def __init__(self, rate):
        """
        Dropout layer

        :param rate: dropout rate (fraction of the input units to drop)
        """
        self.rate = rate
        self.mask = None

    def forward(self, data_input):
        self.input = data_input
        self.mask = np.random.binomial(1, 1 - self.rate, size=self.input.shape)
        self.output = self.input * self.mask
        return self.output

    def backward(self, error, learning_rate):
        return error * self.mask

    def __repr__(self):
        return f'DropoutLayer(rate={self.rate})'

    def __str__(self):
        return f'DropoutLayer(rate={self.rate})'

    def params_count(self):
        return 0

    def summary(self):
        print(f'DropoutLayer: rate = {self.rate}')
