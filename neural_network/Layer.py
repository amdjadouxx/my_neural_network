class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        """
        Forward pass

        :param input: input data

        :return: output
        """
        raise NotImplementedError

    def backward(self, output_error, learning_rate):
        """
        Backward pass

        :param output_error: error of the layer
        :param learning_rate: learning rate

        :return: error of the previous layer
        """
        raise NotImplementedError

    def params_count(self):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError
    
    def __str__(self):
        raise NotImplementedError

    def __doc__(self):
        raise NotImplementedError