import numpy as np
import pickle as pkl
from .FCLayer import FCLayer
from .ActivationFunc import *
from .LossesFunc import *
from .DisplayTrainStats import *
from .DropoutLayer import DropoutLayer
from .FCLayer import FCLayer
from .ActivationLayer import ActivationLayer
from .config import name_to_loss_func


class Network:

    def __init__(self, loss='mse'):
        self.layers = []
        self.err_logs = []
        self.accuracy_logs = []
        (func, func_prime) = name_to_loss_func(loss)
        self.loss = func
        self.loss_prime = func_prime

    def add(self, layer):
        """
        Ajoute une couche au réseau.
        
        : layer : Layer : couche à ajouter
        """
        self.layers.append(layer)

    def predict(self, input_data) -> list:
        """
        Prédit la sortie pour les données d'entrée données.
        
        : input_data : np.array : données d'entrée
        """
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate, silent=False, eval=True, threshold=None, patience=0):
        """
        Entraîne le réseau sur les données d'entraînement avec un système de seuil.
        
        : x_train : np.array : données d'entraînement
        : y_train : np.array : étiquettes d'entraînement
        : epochs : int : nombre d'itérations
        : learning_rate : float : taux d'apprentissage
        : silent : bool : affiche les statistiques d'entrainement
        : eval : bool : évalue le réseau (False pour accélérer l'entraînement)
        : threshold : float : seuil d'arrêt
        : patience : int : nombre d'itérations sans amélioration avant l'arrêt
        """
        self.clear_logs()
        samples = len(x_train)
        patience_counter = 0
        best_loss = float('inf')

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

            if threshold is not None:
                if err < best_loss:
                    best_loss = err
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'Early stopping at epoch {i+1}')
                        break

    def clear_logs(self):
        """
        Efface les statistiques d'entraînement.
        """
        self.err_logs = []
        self.accuracy_logs = []

    def evaluate(self, x_test, y_test, silent=True) -> float:
        """
        Évalue le réseau sur les données de test.

        : x_test : np.array : données de test
        : y_test : np.array : étiquettes de test
        : silent : bool : affiche les statistiques d'entraînement

        : return : float : précision

        """
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
        """
        Affiche un résumé du réseau.
        """
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
        """
        Affiche les statistiques d'entraînement.
        """
        disp_loss_graph(self.err_logs)

    def disp_accuracy_graph(self):
        """
        Affiche les statistiques d'entraînement.
        """
        disp_accuracy_graph(self.accuracy_logs)

    def disp_loss_accuracy_graph(self):
        """
        Affiche les statistiques d'entraînement.
        """
        disp_loss_accuracy_graph(self.err_logs, self.accuracy_logs)

    def show(self):
        """
        Affiche les graphiques.
        """
        show()

    def save(self, filename):
        """
        Sauvegarde le réseau dans un fichier.
        """
        with open(filename, 'wb') as file:
            pkl.dump(self, file)

    def load(self, filename):
        """
        Charge un réseau depuis un fichier.
        """
        with open(filename, 'rb') as file:
            loaded_model = pkl.load(file)
        self.layers = loaded_model.layers
        self.loss = loaded_model.loss
        self.loss_prime = loaded_model.loss_prime
        self.err_logs = loaded_model.err_logs
        self.accuracy_logs = loaded_model.accuracy_logs

    def confusion_matrix(self, x_test, y_test):
        """
        Affiche la matrice de confusion pour les problèmes de classification.

        Score_f1 is the weighted average of Precision and Recall.

        Precision = TP / (TP + FP) is the ratio of correctly predicted positive observations to the total predicted positives.

        Recall = TP / (TP + FN) is the ratio of correctly predicted positive observations to the all observations in actual class.

        : x_test : np.array : données de test
        : y_test : np.array : étiquettes de test

        : return : np.array : matrice de confusion
        """
        num_classes = y_test.shape[1]
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        
        for i in range(len(x_test)):
            output = self.predict([x_test[i]])[0]

            predicted = np.argmax(output)
            expected = np.argmax(y_test[i])
            
            confusion_matrix[expected][predicted] += 1
        
        TP = np.diag(confusion_matrix)
        FP = np.sum(confusion_matrix, axis=0) - TP
        FN = np.sum(confusion_matrix, axis=1) - TP
        TN = np.sum(confusion_matrix) - (FP + FN + TP)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall)

        print("Metrics by class:")
        for i in range(len(TP)):
            print(f"=====================================")
            print(f"Class {i}:")
            print(f"  Precision: {precision[i]:.4f}")
            print(f"  Recall: {recall[i]:.4f}")
            print(f"  F1 Score: {f1_score[i]:.4f}")
            print(f"=====================================")


        return confusion_matrix
