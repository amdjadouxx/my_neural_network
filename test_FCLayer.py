import unittest
import numpy as np
from Neuron import Neuron
from FCLayer import FCLayer
from ActivationFunctions import sigmoid, sigmoid_derivative


class TestFCLayer(unittest.TestCase):
    def setUp(self):
        """
        Set up the test
        """
        self.input_size = 2
        self.output_size = 3
        self.activation = sigmoid
        self.activation_derivative = sigmoid_derivative
        self.layer = FCLayer(self.input_size, self.output_size, self.activation, self.activation_derivative)

    def test_forward(self):
        """
        Test the forward pass
        """
        inputs = np.array([1, 2])
        output = self.layer.forward(inputs)
        expected_output = [neuron.forward(inputs) for neuron in self.layer.neurons]
        self.assertTrue(np.allclose(output, expected_output, atol=1e-5))

    def test_backward(self):
        """
        Test the backward pass
        """
        inputs = np.array([1, 2])
        self.layer.forward(inputs)
        errors = [0.5, 0.3, 0.1]
        backward_output = self.layer.backward(errors)
        expected_output = [neuron.backward(error) for neuron, error in zip(self.layer.neurons, errors)]
        self.assertTrue(np.allclose(backward_output, expected_output, atol=1e-5))

    def test_update(self):
        """
        Test the update method
        """
        weights = [np.array([0.5, -0.5]), np.array([0.3, -0.3]), np.array([0.1, -0.1])]
        biases = [0.1, 0.2, 0.3]
        self.layer.update(weights, biases)
        for neuron, weight, bias in zip(self.layer.neurons, weights, biases):
            self.assertTrue(np.array_equal(neuron.weights, weight))
            self.assertEqual(neuron.bias, bias)

    def test_update_weights(self):
        """
        Test the update weights method
        """
        inputs = np.array([1.0, 2.0])
        self.layer.forward(inputs)
        errors = [0.5, 0.3, 0.1]
        learning_rate = 0.01

        expected_weights = []
        expected_biases = []
        for neuron, error in zip(self.layer.neurons, errors):
            expected_weights.append(neuron.weights - learning_rate * error * inputs)
            expected_biases.append(neuron.bias - learning_rate * error)

        self.layer.update_weights(learning_rate, errors)

        for neuron, exp_weights, exp_bias in zip(self.layer.neurons, expected_weights, expected_biases):

            self.assertTrue(np.allclose(neuron.weights, exp_weights, atol=1e-5))
            self.assertAlmostEqual(neuron.bias, exp_bias, places=5)

    def test_clipping_weights(self):
        """
        Test clipping the weights
        """
        self.layer.clipping_weights()
        for neuron in self.layer.neurons:
            self.assertTrue(np.all(neuron.weights >= -1))
            self.assertTrue(np.all(neuron.weights <= 1))

    def test_get_weights(self):
        """
        Test the get weights method
        """
        weights = self.layer.get_weights()
        for neuron, weight in zip(self.layer.neurons, weights):
            self.assertTrue(np.array_equal(neuron.weights, weight))

    def test_get_biases(self):
        """
        Test the get biases method
        """
        biases = self.layer.get_biases()
        for neuron, bias in zip(self.layer.neurons, biases):
            self.assertEqual(neuron.bias, bias)

    def test_set_weights(self):
        """
        Test the set weights method
        """
        weights = [np.array([0.5, -0.5]), np.array([0.3, -0.3]), np.array([0.1, -0.1])]
        self.layer.set_weights(weights)
        for neuron, weight in zip(self.layer.neurons, weights):
            self.assertTrue(np.array_equal(neuron.weights, weight))

    def test_set_biases(self):
        """
        Test the set biases method
        """
        biases = [0.1, 0.2, 0.3]
        self.layer.set_biases(biases)
        for neuron, bias in zip(self.layer.neurons, biases):
            self.assertEqual(neuron.bias, bias)

if __name__ == "__main__":
    unittest.main()