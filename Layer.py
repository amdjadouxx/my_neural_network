class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_error, learning_rate):
        raise NotImplementedError

    def params_count(self):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError