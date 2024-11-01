import numpy as np
from Layer import Layer

class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.biases = np.random.rand(1, output_size) - 0.5

    def forward(self, data_input):
        """
        Forward pass

        :param data_input: input data
        :return: output
        """
        self.input = data_input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward(self, error, learning_rate):
        """
        Backward pass

        :param error: error of the layer
        :param learning_rate: learning rate
        :return: error of the previous layer
        """
        weights_error = np.dot(self.input.T, error)

        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * error

        return np.dot(error, self.weights.T)

    def __repr__(self):
        return f'FCLayer: {self.weights.shape[0]} inputs, {self.weights.shape[1]} outputs'

    def __str__(self):
        return f'FCLayer: {self.weights.shape[0]} inputs, {self.weights.shape[1]} outputs'

    def params_count(self):
        return self.weights.size + self.biases.size

    def summary(self):
        self.__str__()
        print(f'Paramètres: {self.params_count()}')