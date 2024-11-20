import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QLineEdit, QComboBox, QFileDialog, QTextEdit
from neural_network.Network import Network
from neural_network.FCLayer import FCLayer
from neural_network.ActivationLayer import ActivationLayer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Neural Network Builder")
        self.setGeometry(100, 100, 800, 600)  # Set window size to 800x600

        self.layout = QVBoxLayout()

        self.loss_function = QComboBox()
        self.loss_function.addItems(["mse", "cross_entropy"])
        self.layout.addWidget(QLabel("Select Loss Function"))
        self.layout.addWidget(self.loss_function)

        self.layer_type = QComboBox()
        self.layer_type.addItems(["Fully Connected Layer", "Activation Layer"])
        self.layout.addWidget(QLabel("Select Layer Type"))
        self.layout.addWidget(self.layer_type)

        self.input_size = QLineEdit()
        self.input_size.setPlaceholderText("Input Size")
        self.layout.addWidget(self.input_size)

        self.output_size = QLineEdit()
        self.output_size.setPlaceholderText("Output Size")
        self.layout.addWidget(self.output_size)

        self.activation_function = QLineEdit()
        self.activation_function.setPlaceholderText("Activation Function (e.g., 'tanh', 'sigmoid')")
        self.layout.addWidget(self.activation_function)

        self.add_layer_button = QPushButton("Add Layer")
        self.add_layer_button.clicked.connect(self.add_layer)
        self.layout.addWidget(self.add_layer_button)

        self.save_button = QPushButton("Save Model to File")
        self.save_button.clicked.connect(self.save_model_to_file)
        self.layout.addWidget(self.save_button)

        self.network_display = QTextEdit()
        self.network_display.setReadOnly(True)
        self.layout.addWidget(self.network_display)

        self.container = QWidget()
        self.container.setLayout(self.layout)
        self.setCentralWidget(self.container)

        self.network = Network()
        self.layers_code = []

    def add_layer(self):
        layer_type = self.layer_type.currentText()
        input_size = int(self.input_size.text())
        output_size = int(self.output_size.text())
        activation_function = self.activation_function.text()

        if layer_type == "Fully Connected Layer":
            self.network.add(FCLayer(input_size, output_size))
            self.layers_code.append(f"net.add(FCLayer({input_size}, {output_size}))")
        elif layer_type == "Activation Layer":
            self.network.add(ActivationLayer(activation_function))
            self.layers_code.append(f"net.add(ActivationLayer('{activation_function}'))")

        self.update_network_display()

    def update_network_display(self):
        display_text = "Network Layers:\n"
        for layer in self.layers_code:
            display_text += f"{layer}\n"
        self.network_display.setText(display_text)

    def save_model_to_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "Python Files (*.py);;All Files (*)", options=options)
        if file_path:
            with open(file_path, 'w') as file:
                file.write(self.generate_model_code())
            self.network_display.setText(f"Model saved to {file_path}")

    def generate_model_code(self):
        loss_function = self.loss_function.currentText()
        code = [
            "import numpy as np",
            "from neural_network.Network import Network",
            "from neural_network.FCLayer import FCLayer",
            "from neural_network.ActivationLayer import ActivationLayer",
            "",
            "def create_model():",
            f"    net = Network('{loss_function}')"
        ]
        code.extend([f"    {layer}" for layer in self.layers_code])
        code.append("    return net")
        return "\n".join(code)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())