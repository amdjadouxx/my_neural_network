import numpy as np


def mse(y_true, y_pred):
    """
    Mean squared error

    :param y_true: true values
    :param y_pred: predicted values
    :return: mean squared error
    """
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    """
    Derivative of the mean squared error

    :param y_true: true values
    :param y_pred: predicted values
    :return: derivative of the mean squared error
    """
    return 2 * (y_pred - y_true) / y_true.size