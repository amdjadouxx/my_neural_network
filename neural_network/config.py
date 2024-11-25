from .LossesFunc import *
from .ActivationFunc import *
from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QLineEdit, QComboBox, QFileDialog, QApplication, QDialog, QDialogButtonBox, QMessageBox, QListWidget
from .DropoutLayer import DropoutLayer
from .ConvLayer import ConvLayer
from .FCLayer import FCLayer
from .FlattenLayer import FlattenLayer


losses_func_dict = {
    'mse' : (mse, mse_prime),
    'cross_entropy' : (cross_entropy, cross_entropy_prime)
    }

activation_func_dict = {
    'sigmoid' : (sigmoid, sigmoid_prime),
    'tanh' : (tanh, tanh_prime),
    'step' : (step, step_prime),
    'relu' : (relu, relu_prime),
    'softmax' : (softmax, softmax_prime)
    }

#ex: name of the layer get by __str__ method : (file name, class name)
layer_types = {
    str(FCLayer.__name__) : ('FCLayer', 'FCLayer'),
    'Activation Layer': ('ActivationLayer', 'ActivationLayer'),
    str(ConvLayer.__name__) : ('ConvLayer', 'ConvLayer'),
    str(FlattenLayer.__name__) : ('FlattenLayer', 'FlattenLayer'),
    str(DropoutLayer.__name__) : ('DropoutLayer', 'DropoutLayer')
    }

def name_to_loss_func(name):
    """
    Get the loss function by its name

    :param name: str: name of the function

    
    """
    return losses_func_dict[name]

def name_to_activation_func(name):
    """
    Get the activation function by its name

    :param name: str: name of the function
    """
    return activation_func_dict[name]
