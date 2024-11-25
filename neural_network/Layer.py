class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        """
        Forward pass

        :param input: np.array: input data

        :return: np.array: output data
        """
        raise NotImplementedError

    def backward(self, output_error, learning_rate):
        """
        Backward pass

        :param error: np.array: error
        :param learning_rate: float: learning rate
        :return: np.array: new error
        """
        raise NotImplementedError

    def params_count(self):
        """
        Count the number of parameters of the layer.

        :return: int: number of parameters
        """
        raise NotImplementedError

    def summary(self):
        """
        Display a summary of the layer.
        """
        raise NotImplementedError
    
    def __str__(self):
        """
        Display the type of the layer.

        :return: str: name of the layer

        """
        raise NotImplementedError

    def __doc__(self):
        """
        Display the description of the layer.
        
        :return: str: description of the layer
        """
        raise NotImplementedError
    
    def get_load_line(self):
        """
        Get the line of code to load the layer.

        :return: str: line of code
        """
        raise NotImplementedError