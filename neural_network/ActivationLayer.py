import numpy as np
from .ActivationFunc import *
from .Layer import Layer
from .config import name_to_activation_func

class ActivationLayer(Layer):

    def __init__(self, activation):
        func, func_prime = name_to_activation_func(activation)
        self.activation = func
        self.activation_prime = func_prime

    def forward(self, data_input):
        self.input = data_input
        self.output = self.activation(self.input)
        return self.output

    def backward(self, error, learning_rate):
        return error * self.activation_prime(self.input)

    def __str__(self):
        return f'ActivationLayer: {self.activation}'

    def params_count(self):
        return 0

    def summary(self):
        print(f'ActivationLayer: function = {self.activation.__name__} with {self.input.size} inputs')

    def __doc__(self):
        return 'Activation layer used in neural networks to apply a function to the input data to detect complex patterns'