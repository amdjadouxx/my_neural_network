from .LossesFunc import *
from .ActivationFunc import *

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

#ex: name of the layer : (file name, class name)
layer_types = {
    'Fully Connected Layer' : ('FCLayer', 'FCLayer'),
    'Activation Layer' : ('ActivationLayer', 'ActivationLayer'),
    'Convolutional Layer' : ('ConvLayer', 'ConvLayer'),
    'Flatten Layer' : ('FlattenLayer', 'FlattenLayer'),
    'Dropout Layer' : ('DropoutLayer', 'DropoutLayer')
    }

def name_to_loss_func(name):
    """
    Get the loss function by its name
    """
    return losses_func_dict[name]

def name_to_activation_func(name):
    """
    Get the activation function by its name
    """
    return activation_func_dict[name]
