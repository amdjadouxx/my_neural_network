import unittest
import numpy as np
from Neuron import Neuron
from ActivationFunctions import sigmoid, sigmoid_derivative

class TestNeuron(unittest.TestCase):
    def setUp(self):
        self.weights = np.array([0.5, -0.5])
        self.bias = 0.1
        self.neuron = Neuron(self.weights.copy(), self.bias, sigmoid, sigmoid_derivative)

    def test_forward(self):
        inputs = np.array([1, 2])
        output = self.neuron.forward(inputs)
        expected_output = sigmoid(np.dot(self.weights, inputs) + self.bias)
        self.assertAlmostEqual(output, expected_output, places=5)

    def test_update(self):
        new_weights = np.array([0.3, -0.3])
        new_bias = 0.2
        self.neuron.update(new_weights, new_bias)
        self.assertTrue(np.array_equal(self.neuron.weights, new_weights))
        self.assertEqual(self.neuron.bias, new_bias)

    def test_backward(self):
        inputs = np.array([1, 2])
        self.neuron.forward(inputs)
        error = 0.5
        backward_output = self.neuron.backward(error)
        expected_output = sigmoid_derivative(self.neuron.output) * error
        self.assertAlmostEqual(backward_output, expected_output, places=5)

    def test_update_weights(self):
        inputs = np.array([1.0, 2.0])
        self.neuron.forward(inputs)
        error = 0.5
        learning_rate = 0.01
        self.neuron.update_weights(learning_rate, error)

        expected_weights = self.weights - learning_rate * error * inputs
        expected_bias = self.bias - learning_rate * error

        print("Actual weights:", self.neuron.weights)
        print("Expected weights:", expected_weights)
        self.assertTrue(np.allclose(self.neuron.weights, expected_weights, atol=1e-5))
        self.assertAlmostEqual(self.neuron.bias, expected_bias, places=5)

if __name__ == '__main__':
    unittest.main()