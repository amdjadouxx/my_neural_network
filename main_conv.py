import numpy as np
import os
import matplotlib.pyplot as plt
from neural_network.Network import Network
from neural_network.FCLayer import FCLayer
from neural_network.ConvLayer import ConvLayer
from neural_network.FlattenLayer import FlattenLayer
from neural_network.ActivationLayer import ActivationLayer

from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_train /= 255
y_train = to_categorical(y_train)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32')
x_test /= 255
y_test = to_categorical(y_test)

def load():
    net = Network()
    net.load('mnist_conv.pkl')
    return net

def save():
    net = Network('mse')
    net.add(ConvLayer((28, 28, 1), (3, 3), 1))
    net.add(ActivationLayer('tanh'))
    net.add(FlattenLayer())
    net.add(FCLayer(26*26*1, 100))
    net.add(ActivationLayer('tanh'))
    net.add(FCLayer(100, 10))
    net.add(ActivationLayer('sigmoid'))
    net.fit(x_train[0:3000], y_train[0:3000], epochs=1000, learning_rate=0.1, threshold=0.01, patience=5, eval=False)
    net.save('mnist_conv.pkl')
    return net

if __name__ == '__main__':
    if os.path.exists('mnist_conv.pkl'):
        net = load()
    else:
        net = save()

    net.evaluate(x_test[0:1000], y_test[0:1000], silent=False)
    net.summary()
    #predict a png image
    img = plt.imread('pixil-frame-2.png')
    img = img[:,:,0]
    img = img.reshape(1, 28, 28, 1)
    out = net.predict(img)
    print(np.argmax(out))