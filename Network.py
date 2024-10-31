import numpy as np
from FCLayer import FCLayer
from ActivationFunc import *
from LossesFunc import *

class Network:
    def __init__(self, loss=mse, loss_prime=mse_prime):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        """Ajoute une couche au réseau."""
        self.layers.append(layer)

    def predict(self, input_data):
        """Prédit la sortie pour les données d'entrée données."""
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        """Entraîne le réseau sur les données d'entraînement."""
        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward(output)

                err += self.loss(y_train[j], output)

                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)

            err /= samples
            print(f'epoch {i+1}/{epochs} loss = {err}')

    def summary(self):
        """Affiche un résumé du réseau."""
        print('Résumé du réseau')
        print('========================================')
        print('Layer (type)                 Output Shape              Param #   ')
        print('========================================')
        total_params = 0
        for layer in self.layers:
            layer.summary()
            total_params += layer.params_count()
        print('========================================')
        print(f'Paramètres totaux: {total_params}')
        print('========================================')

