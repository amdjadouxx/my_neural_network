from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QLineEdit, QComboBox, QFileDialog, QTextEdit, QApplication
import sys
from .Network import Network
from .FCLayer import FCLayer
from .ActivationLayer import ActivationLayer
from .config import losses_func_dict, activation_func_dict

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Neural Network Builder")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()

        self.loss_function = QComboBox()
        self.loss_function.addItems(losses_func_dict.keys())
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
        self.activation_function.setPlaceholderText(f"Activation Function (Options: {', '.join(activation_func_dict.keys())})")
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
        input_size = None
        output_size = None
        activation_function = None
        if layer_type == "Fully Connected Layer":
            input_size = int(self.input_size.text())
            output_size = int(self.output_size.text())
        elif layer_type == "Activation Layer":
            activation_function = self.activation_function.text()

        if layer_type == "Fully Connected Layer" and input_size and output_size:
            self.network.add(FCLayer(input_size, output_size))
            self.layers_code.append(f"net.add(FCLayer({input_size}, {output_size}))")
        elif layer_type == "Activation Layer" and activation_function:
            self.network.add(ActivationLayer(activation_function))
            self.layers_code.append(f"net.add(ActivationLayer('{activation_function}'))")

        self.update_network_display()
        self.clear_input_fields()

    def update_network_display(self):
        display_text = "Network Layers:\n"
        for layer in self.layers_code:
            display_text += f"{layer}\n"
        self.network_display.setText(display_text)

    def clear_input_fields(self):
        self.input_size.clear()
        self.output_size.clear()
        self.activation_function.clear()

    def save_model_to_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "Python Files (*.py);;All Files (*)", options=options)
        if file_path:
            self.network.save_model_to_file(file_path, self.loss_function.currentText(), self.layers_code)
            self.network_display.setText(f"Model saved to {file_path}")

    def show_windows():
        pass

def show_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
