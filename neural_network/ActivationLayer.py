import numpy as np
from .ActivationFunc import *
from .Layer import Layer

class ActivationLayer(Layer):

    def __init__(self, activation):
        func, func_prime = self.func_to_name(activation)
        self.activation = func
        self.activation_prime = func_prime

    def func_to_name(self, func):
        if func == 'tanh':
            return tanh, tanh_prime
        elif func == 'sigmoid':
            return sigmoid, sigmoid_prime
        elif func == 'step':
            return step, step_prime

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
        print(f'ActivationLayer: function = {self.activation.__name__} with {self.input.size} inputs')