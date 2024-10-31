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

def step_prime(x):
    """
    Derivative of the step activation function

    rules : the derivative is 0

    :param x: input
    :return: output
    """
    return 0