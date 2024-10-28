import numpy as np

def sigmoid(x):
    """
    Sigmoid activation function

    :param x: input
    :return: output
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Derivative of the sigmoid activation function

    :param x: input
    :return: output
    """
    return sigmoid(x) * (1 - sigmoid(x))