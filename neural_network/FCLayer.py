import numpy as np
from .Layer import Layer

class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.random.rand(1, output_size) - 0.5

    def forward(self, data_input) -> np.ndarray:
        self.input = data_input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward(self, error, learning_rate) -> np.ndarray:
        weights_error = np.dot(self.input.T, error)

        self.weights -= learning_rate * weights_error
        self.biases -= learning_rate * error

        return np.dot(error, self.weights.T)

    def __str__(self) -> str:
        return f'FCLayer: {self.weights.shape[0]} inputs, {self.weights.shape[1]} outputs'

    def params_count(self) -> int:
        return self.weights.size + self.biases.size

    def summary(self):
        self.__str__()
        print(f'Paramètres: {self.params_count()}')

    def __doc__(self) -> str:
        return 'Fully connected layer used in neural networks to weight inputs and add biases to them'
