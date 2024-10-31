import numpy as np
from ActivationFunc import *
from Layer import Layer

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, data_input):
        self.input = data_input
        self.output = self.activation(self.input)
        return self.output

    def backward(self, error, learning_rate):
        return error * self.activation_prime(self.input)

    def __repr__(self):
        return f'ActivationLayer: {self.activation}'

    def __str__(self):
        return f'ActivationLayer: {self.activation}'

    def params_count(self):
        return 0

    def summary(self):
        print(f'ActivationLayer: function = {self.activation} and derivative = {self.activation_prime} with {self.input.size} inputs')