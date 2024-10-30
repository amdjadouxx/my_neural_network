from Neuron import Neuron
import numpy as np

class FCLayer:
    def __init__(self, input_size, output_size, activation, activation_derivative):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.neurons = [
            Neuron(
                weights=np.random.randn(input_size),
                bias=np.random.randn(),
                activation_function=activation,
                activation_function_derivative=activation_derivative
            ) for _ in range(output_size)
        ]

    def forward(self, inputs):
        return [neuron.forward(inputs) for neuron in self.neurons]

    def backward(self, errors):
        return [neuron.backward(error) for neuron, error in zip(self.neurons, errors)]

    def update(self, weights, biases):
        for neuron, weight, bias in zip(self.neurons, weights, biases):
            neuron.update(weight, bias)

    def update_weights(self, learning_rate, errors):
        for neuron, error in zip(self.neurons, errors):
            neuron.update_weights(learning_rate, error)

    def clipping_weights(self):
        for neuron in self.neurons:
            neuron.clipping_weights()

    def get_weights(self):
        return [neuron.weights for neuron in self.neurons]

    def get_biases(self):
        return [neuron.bias for neuron in self.neurons]

    def set_weights(self, weights):
        for neuron, weight in zip(self.neurons, weights):
            neuron.weights = weight

    def set_biases(self, biases):
        for neuron, bias in zip(self.neurons, biases):
            neuron.bias = bias