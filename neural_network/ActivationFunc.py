import numpy as np

def sigmoid(x):
    """
    Sigmoid activation function

    rules:  the output is close to 1 when x is higher than 0
            the output is close to 0 when x is lower than 0
            the output is 0.5 when x is equal to 0

    :param x: input
    :return: output
    """
    return 1 / (1 + np.exp(-x))

sigmoid.__name__ = "sigmoid"

def sigmoid_prime(x):
    """
    Derivative of the sigmoid activation function

    rules : when x is higher than 0, the derivative is positive
            when x is lower than 0, the derivative is negative
            when x is equal to 0, the derivative is 0

    :param x: input
    :return: output
    """
    return sigmoid(x) * (1 - sigmoid(x))

sigmoid_prime.__name__ = "sigmoid_prime"

def tanh(x):
    """
    Hyperbolic tangent activation function

    rules:  the output is close to 1 when x is higher than 0
            the output is close to -1 when x is lower than 0
            the output is 0 when x is equal to 0

    :param x: input
    :return: output
    """
    return np.tanh(x)

tanh.__name__ = "tanh"

def tanh_prime(x):
    """
    Derivative of the hyperbolic tangent activation function

    rules : when x is higher than 0, the derivative is positive
            when x is lower than 0, the derivative is negative
            when x is equal to 0, the derivative is 0

    :param x: input
    :return: output
    """
    return 1 - np.tanh(x) ** 2

tanh_prime.__name__ = "tanh_prime"

def step(x):
    """
    Step activation function

    rules:  the output is 1 when x is higher than 0
            the output is 0 when x is lower than 0
            the output is 1 when x is equal to 0

    :param x: input
    :return: output
    """
    return np.heaviside(x, 1)

step.__name__ = "step"

def step_prime(x):
    """
    Derivative of the step activation function

    rules : the derivative is 0

    :param x: input
    :return: output
    """
    return 0

step_prime.__name__ = "step_prime"

def relu(x):
    """
    Rectified linear unit activation function

    rules:  the output is equal to x when x is higher than 0
            the output is 0 when x is lower than 0

    :param x: input
    :return: output
    """
    return np.maximum(0, x)

relu.__name__ = "relu"

def relu_prime(x):
    """
    Derivative of the rectified linear unit activation function

    rules : the derivative is 1 when x is higher than 0
            the derivative is 0 when x is lower than 0

    :param x: input
    :return: output
    """
    return np.heaviside(x, 1)
