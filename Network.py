import numpy as np
import pickle as pkl
from FCLayer import FCLayer
from ActivationFunc import *
from LossesFunc import *
from DisplayTrainStats import *
from DropoutLayer import DropoutLayer

class Network:
    def __init__(self, loss=mse, loss_prime=mse_prime):
        self.layers = []
        self.err_logs = []
        self.accuracy_logs = []
        self.loss = loss
        self.loss_prime = loss_prime

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

    def fit(self, x_train, y_train, epochs, learning_rate, silent=False, eval=True):
        """Entraîne le réseau sur les données d'entraînement."""
        self.clear_logs()
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
            if not silent:
                print(f'epoch {i+1}/{epochs} loss = {err}')
            self.err_logs.append(err)
            if eval:
                self.accuracy_logs.append(self.evaluate(x_train, y_train))

    def clear_logs(self):
        """Efface les statistiques d'entraînement."""
        self.err_logs = []
        self.accuracy_logs = []

    def evaluate(self, x_test, y_test, silent=True):
        """Évalue le réseau sur les données de test."""
        samples = len(x_test)
        correct = 0

        for i in range(samples):
            output = x_test[i]
            for layer in self.layers:
                output = layer.forward(output)

            predicted = output
            expected = y_test[i]

            if np.argmax(predicted) == np.argmax(expected):
                correct += 1

        accuracy = correct / samples
        if not silent:
            print(f'Accuracy = {accuracy * 100}%')
        return accuracy

    def summary(self):
        """Affiche un résumé du réseau."""
        print('Résumé du réseau')
        print('========================================')
        print('Fonction de perte: ', self.loss.__name__)
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

    def disp_loss_graph(self):
        """Affiche les statistiques d'entraînement."""
        disp_loss_graph(self.err_logs)

    def disp_accuracy_graph(self):
        """Affiche les statistiques d'entraînement."""
        disp_accuracy_graph(self.accuracy_logs)

    def disp_loss_accuracy_graph(self):
        """Affiche les statistiques d'entraînement."""
        disp_loss_accuracy_graph(self.err_logs, self.accuracy_logs)

    def show(self):
        """Affiche les graphiques."""
        show()

    def save(self, filename):
        """Sauvegarde le réseau dans un fichier."""
        with open(filename, 'wb') as file:
            pkl.dump(self, file)

    def load(self, filename):
        """Charge un réseau depuis un fichier."""
        with open(filename, 'rb') as file:
            loaded_model = pkl.load(file)
        self.layers = loaded_model.layers
        self.loss = loaded_model.loss
        self.loss_prime = loaded_model.loss_prime
        self.err_logs = loaded_model.err_logs
        self.accuracy_logs = loaded_model.accuracy_logs