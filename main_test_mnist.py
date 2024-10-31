import numpy as np
from FCLayer import FCLayer
from ActivationLayer import ActivationLayer
from ActivationFunc import *
from LossesFunc import *
from Network import Network
import sys

from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255

y_train = to_categorical(y_train)

x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = to_categorical(y_test)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main_test_mnist.py [train/test]")
        sys.exit(84)

    if sys.argv[1] == "train":
        net = Network(loss=mse, loss_prime=mse_prime)
        net.add(FCLayer(28*28, 100))
        net.add(ActivationLayer(tanh, tanh_prime))
        net.add(FCLayer(100, 50))
        net.add(ActivationLayer(tanh, tanh_prime))
        net.add(FCLayer(50, 10))
        net.add(ActivationLayer(tanh, tanh_prime))
        net.fit(x_train[0:1000], y_train[0:1000], epochs=100, learning_rate=0.1, silent=False)

        net.summary()

        acc = net.evaluate(x_train[0:1000], y_train[0:1000], silent=False)

        if acc > 0.9:
            net.save("mnist_model")
        net.disp_loss_accuracy_graph()
        net.show()

    elif sys.argv[1] == "test":
        net = Network()
        net.load("mnist_model")
        net.summary()

        net.evaluate(x_train[0:1000], y_train[0:1000], silent=False)

    else:
        print("Invalid argument")
        sys.exit(84)

    out = net.predict(x_test[0:3])
    for i in range(3):
        print("predicted value : ", np.argmax(out[i]), end="\n")
        print("true value : ", np.argmax(y_test[i]), end="\n")
