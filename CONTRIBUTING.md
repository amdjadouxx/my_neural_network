# Contributing to My Neural Network Project

Thank you for considering contributing to this project! Here are some guidelines to help you get started.

## Adding a Layer

To add a new layer to the project, follow these steps:

1. Create a new file for your layer in the `neural_network` directory.
2. Define your layer class by inheriting from the `Layer` base class.
3. Implement the required methods (`forward`, `backward`, `params_count`, etc.).
4. Add your layer to the `__init__.py` file in the `neural_network` directory to make it importable.

Example:
```python
# filepath: neural_network/MyNewLayer.py
from .Layer import Layer

class MyNewLayer(Layer):
    def __init__(self, ...):
        # ...initialize your layer...
    
    def forward(self, input_data):
        # ...forward pass logic...
    
    def backward(self, error, learning_rate):
        # ...backward pass logic...
    
    def params_count(self):
        # ...return the number of parameters...

    #etc....
```

## Adding an Activation Function

1. Define the activation function and its derivative in the ActivationFunc.py file.
2. Update the name_to_activation_func function in the config.py file to include your new activation function

Example:
```python
# filepath: neural_network/ActivationFunc.py
def my_activation(x):
    # ...activation function logic...

def my_activation_prime(x):
    # ...derivative logic...

# filepath: neural_network/config.py
activation_func_dict = {
    'sigmoid' : (sigmoid, sigmoid_prime),
    ...
    'YOUR_FUNC' : (my_activation, my_activation_prime)
    }

```

## Adding a Loss Function

1. Define the activation function and its derivative in the ActivationFunc.py file.
2. Update the name_to_activation_func function in the config.py file to include your new activation function

Example:
```python
# filepath: neural_network/LossesFunc.py
def my_loss(y_true, y_pred):
    # ...loss function logic...

def my_loss_prime(y_true, y_pred):
    # ...derivative logic...

# filepath: neural_network/config.py
losses_func_dict = {
    'mse' : (mse, mse_prime),
    'cross_entropy' : (cross_entropy, cross_entropy_prime),
    ...
    'YOUR_FUNC' : (my_activation, my_activation_prime)
    }
```

## Adding a Method to the Network Class

1. Define your method in the Network.py file.
2. Update the HELP.md file to document our new method

Exemple:
````python
# filepath: neural_network/Network.py
class Network:
    # ...existing code...
    
    def my_new_method(self, ...):
        # ...method logic...

## Important Notes
1. Ensure your code follows the project's coding standards.
2. Write tests for your new features and ensure all existing tests pass.
3. Update the documentation as needed.

```

Thank you for your contributions!