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
        """
        Create a neural network.

        :loss: str: loss function name (mse, cross_entropy)
        """
        self.layers = []
        self.err_logs = []
        self.accuracy_logs = []
        (func, func_prime) = name_to_loss_func(loss)
        self.loss = func
        self.loss_prime = func_prime

    def add(self, layer):
        """
        Add a layer to the network.

        :layer: Layer: layer to add
        """
        self.layers.append(layer)

    def predict(self, input_data) -> list:
        """
        Predict the output of the network.

        :input_data: np.array: input data
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
        Train the network.

        :x_train: np.array: training data
        :y_train: np.array: training labels
        :epochs: int: number of epochs
        :learning_rate: float: learning rate
        :silent: bool: display training statistics
        :eval: bool: evaluate the network on the training data (desactivate to speed up training)
        :threshold: float: early stopping threshold
        :patience: int: early stopping patience
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
        Clear the logs.
        """
        self.err_logs = []
        self.accuracy_logs = []

    def evaluate(self, x_test, y_test, silent=True) -> float:
        """
        Evaluate the network on a dataset.

        :x_test: np.array: test data
        :y_test: np.array: test labels
        :silent: bool: display evaluation statistics

        :return: float: accuracy
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
        Print a summary of the network.
        """
        print('Summary: ')
        print('========================================')
        print('Loss function: ', self.loss.__name__)
        print('========================================')
        print('Layer (type)                 Output Shape              Param #   ')
        print('========================================')
        total_params = 0
        for layer in self.layers:
            layer.summary()
            total_params += layer.params_count()
        print('========================================')
        print(f'Total params: {total_params}')
        print('========================================')

    def disp_loss_graph(self):
        """
        Display the training statistics
        """
        disp_loss_graph(self.err_logs)

    def disp_accuracy_graph(self):
        """
        Display the training statistics
        """
        disp_accuracy_graph(self.accuracy_logs)

    def disp_loss_accuracy_graph(self):
        """
        Display the training statistics
        """
        disp_loss_accuracy_graph(self.err_logs, self.accuracy_logs)

    def show(self):
        """
        Display the training statistics
        """
        show()

    def save(self, filename):
        """
        Save the network to a file.
        """
        with open(filename, 'wb') as file:
            pkl.dump(self, file)

    def load(self, filename):
        """
        Load the network from a file
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
        Generate the confusion matrix for the network.

        :x_test: np.array: test data
        :y_test: np.array: test labels

        :return: np.array: confusion matrix

        HOW TO READ THE CONFUSION MATRIX:
        - The diagonal elements represent the number of points for which the predicted label is equal to the true label
        - The off-diagonal elements are those that are misclassified by the model TP/FN | FP/TN
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

    def generate_model_code(self, loss_function, layers_code):
        code = [
            "import numpy as np",
            "from neural_network.Network import Network",
            "from neural_network.FCLayer import FCLayer",
            "from neural_network.ActivationLayer import ActivationLayer",
            "",
            "def create_model():",
            f"    net = Network('{loss_function}')"
        ]
        code.extend([f"    {layer}" for layer in layers_code])
        code.append("    return net")
        return "\n".join(code)

    def save_model_to_file(self, file_path, loss_function, layers_code):
        with open(file_path, 'w') as file:
            file.write(self.generate_model_code(loss_function, layers_code))
