import numpy as np

class Neuron:
    def __init__(self, weights, bias, activation_function, activation_function_derivative):
        """
        Neuron class

        :param weights: weights
        :param bias: bias
        :param activation_function: activation function
        :param activation_function_derivative: derivative of the activation function
        """
        self.weights = weights
        self.bias = bias
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    def forward(self, inputs):
        """
        Forward pass

        :param inputs: inputs
        :return: output
        """
        self.inputs = inputs
        self.output = self.activation_function(np.dot(self.weights, inputs) + self.bias)
        return self.output

    def update(self, weights, bias):
        """
        Update weights and bias

        :param weights: new weights
        :param bias: new bias
        """
        self.weights = weights
        self.bias = bias

    def backward(self, error):
        """
        Backward pass

        :param error: error
        :return: error
        """
        return self.activation_function_derivative(self.output) * error

    def update_weights(self, learning_rate, error):
        """
        Update weights

        :param learning_rate: learning rate
        :param error: error
        """
        self.weights -= learning_rate * error * self.inputs
        self.bias -= learning_rate * error