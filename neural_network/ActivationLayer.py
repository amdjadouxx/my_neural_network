import numpy as np
from .ActivationFunc import *
from .Layer import Layer
from .config import name_to_activation_func

class ActivationLayer(Layer):

    def __init__(self, activation):
        """
        Create an activation layer.

        :activation: str: activation function name (sigmoid, tanh, relu, step)
        """
        func, func_prime = name_to_activation_func(activation)
        self.activation = func
        self.activation_prime = func_prime

    def forward(self, data_input):
        """
        Forward pass

        :param data_input: np.array: input data
        :return: np.array: output data
        """
        self.input = data_input
        self.output = self.activation(self.input)
        return self.output

    def backward(self, error, learning_rate):
        """
        Backward pass

        :param error: np.array: error from the next layer
        :param learning_rate: float: learning rate
        :return: np.array: error to pass to the previous layer
        """
        return error * self.activation_prime(self.input)

    def __str__(self):
        """
        String representation of the activation layer
        """
        return 'Activation Layer'

    def params_count(self):
        """
        Get the number of parameters in the activation layer
        """
        return 0

    def summary(self):
        """
        Print a summary of the activation layer
        """
        print(f'ActivationLayer: function = {self.activation.__name__} with {self.input.size} inputs')

    def __doc__(self):
        """
        Get the documentation of the activation layer
        """
        return 'Activation layer used in neural networks to apply a function to the input data to detect complex patterns'
    
    def get_load_line(self):
        """
        Get the line of code to load the layer
        """
        return f'ActivationLayer("{self.activation.__name__}")'