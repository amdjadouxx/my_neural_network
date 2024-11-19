from .Layer import Layer


class FlattenLayer(Layer):

    def __init__(self):
        self.input = None
    
    def forward(self, input_data):
        self.input = input_data
        self.output = input_data.flatten().reshape((1,-1))
        return self.output

    def backward(self, output_error, learning_rate):
        return output_error.reshape(self.input.shape)
    
    def params_count(self):
        return 0
    
    def summary(self):
        print("Flatten")

    def __str__(self):
        return "Flatten"
    
    def __doc__(self):
        return "Flatten Layer"