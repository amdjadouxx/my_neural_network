import numpy as np


def mse(y_true, y_pred):
    """
    Mean squared error

    :param y_true: true values
    :param y_pred: predicted values
    :return: mean squared error
    """
    return np.mean(np.power(y_true - y_pred, 2))

mse.__name__ = "mse"


def mse_prime(y_true, y_pred):
    """
    Derivative of the mean squared error

    :param y_true: true values
    :param y_pred: predicted values
    :return: derivative of the mean squared error
    """
    return 2 * (y_pred - y_true) / y_true.size

mse_prime.__name__ = "mse_prime"

def cross_entropy(y_true, y_pred):
    """
    Cross entropy loss function

    :param y_true: true values
    :param y_pred: predicted values
    :return: cross entropy loss
    """
    return -np.sum(y_true * np.log(y_pred))

cross_entropy.__name__ = "cross_entropy"

def cross_entropy_prime(y_true, y_pred):
    """
    Derivative of the cross entropy loss function

    :param y_true: true values
    :param y_pred: predicted values
    :return: derivative of the cross entropy loss
    """
    return -y_true / y_pred

cross_entropy_prime.__name__ = "cross_entropy_prime"
