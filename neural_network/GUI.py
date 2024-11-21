from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QLineEdit, QComboBox, QFileDialog, QTextEdit, QApplication, QHBoxLayout, QGridLayout, QDialog, QDialogButtonBox, QMessageBox
import sys
from .Network import Network
from .FCLayer import FCLayer
from .ActivationLayer import ActivationLayer
from .config import losses_func_dict, activation_func_dict

class LayerConfigDialog(QDialog):
    def __init__(self, layer_type, parent=None):
        super().__init__(parent)
        self.layer_type = layer_type
        self.initUI()

    def initUI(self):
        self.setWindowTitle(f"Configure {self.layer_type}")
        self.layout = QVBoxLayout()

        if self.layer_type == "Fully Connected Layer":
            self.input_size = QLineEdit()
            self.input_size.setPlaceholderText("Input Size")
            self.layout.addWidget(self.input_size)

            self.output_size = QLineEdit()
            self.output_size.setPlaceholderText("Output Size")
            self.layout.addWidget(self.output_size)
        elif self.layer_type == "Activation Layer":
            self.activation_function = QLineEdit()
            self.activation_function.setPlaceholderText(f"Activation Function (Options: {', '.join(activation_func_dict.keys())})")
            self.layout.addWidget(self.activation_function)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

        self.setLayout(self.layout)

    def get_layer_config(self):
        try:
            if self.layer_type == "Fully Connected Layer":
                input_size = self.input_size.text()
                output_size = self.output_size.text()
                if not input_size or not output_size:
                    raise ValueError("Input and output sizes cannot be empty.")
                input_size = int(input_size)
                output_size = int(output_size)
                if input_size <= 0 or output_size <= 0:
                    raise ValueError("Input and output sizes must be positive integers.")
                return {
                    "input_size": input_size,
                    "output_size": output_size
                }
            elif self.layer_type == "Activation Layer":
                activation_function = self.activation_function.text()
                if not activation_function:
                    raise ValueError("Activation function cannot be empty.")
                if activation_function not in activation_func_dict.keys():
                    raise ValueError(f"Invalid activation function. Options are: {', '.join(activation_func_dict.keys())}")
                return {
                    "activation_function": activation_function
                }
        except ValueError as e:
            QMessageBox.critical(self, "Error", str(e))
            return None

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
        dialog = LayerConfigDialog(layer_type, self)
        if dialog.exec_() == QDialog.Accepted:
            config = dialog.get_layer_config()
            if config is None:
                return
            if layer_type == "Fully Connected Layer":
                layer = FCLayer(config["input_size"], config["output_size"])
                self.layers_code.append(f"net.add(FCLayer({config['input_size']}, {config['output_size']}))")
            elif layer_type == "Activation Layer":
                layer = ActivationLayer(config["activation_function"])
                self.layers_code.append(f"net.add(ActivationLayer('{config['activation_function']}'))")
            self.network.add(layer)
            self.update_network_display()

    def update_network_display(self):
        display_text = "<h2>Network Layers:</h2><ul>"
        for layer in self.layers_code:
            display_text += f"<li style='color: #3498DB;'>{layer}</li>"
        display_text += "</ul>"
        self.network_display.setHtml(display_text)

    def save_model_to_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "Python Files (*.py);;All Files (*)", options=options)
        if file_path:
            self.network.save_model_to_file(file_path, self.loss_function.currentText(), self.layers_code)
            self.network_display.setText(f"Model saved to {file_path}")

def show_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())