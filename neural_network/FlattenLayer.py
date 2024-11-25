from .Layer import Layer


class FlattenLayer(Layer):

    def __init__(self):
        """
        Flatten layer
        """
        self.input = None
    
    def forward(self, input_data):
        """
        Forward pass

        :param input_data: np.ndarray: input data

        :return: np.ndarray: output data
        """
        self.input = input_data
        self.output = input_data.flatten().reshape((1,-1))
        return self.output

    def backward(self, output_error, learning_rate):
        """
        Backward pass

        :param output_error: np.ndarray: error data
        :param learning_rate: float: learning rate

        :return: np.ndarray: output error
        """
        return output_error.reshape(self.input.shape)
    
    def params_count(self):
        """
        Number of parameters
        """
        return 0
    
    def summary(self):
        """
        Summary
        """
        print("Flatten")

    def __str__(self):
        """
        Return the name of the layer
        """
        return "Flatten Layer"
    
    def __doc__(self):
        """
        Documentation
        """
        return "Flatten Layer"
    
    def get_load_line(self):
        """
        Load line
        """
        return "FlattenLayer()"
    
FlattenLayer.__name__ = "Flatten Layer"