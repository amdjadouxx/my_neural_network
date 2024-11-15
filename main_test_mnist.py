import numpy as np
import os
import time
import matplotlib.pyplot as plt
from neural_network.Network import Network
from neural_network.FCLayer import FCLayer
from neural_network.ActivationLayer import ActivationLayer
from neural_network.DropoutLayer import DropoutLayer


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

def save():
    start_time = time.time()
    net = Network('mse')
    net.add(FCLayer(28*28, 100))
    net.add(ActivationLayer('tanh'))
    net.add(DropoutLayer(0.5))
    net.add(FCLayer(100, 50))
    net.add(ActivationLayer('tanh'))
    net.add(DropoutLayer(0.5))
    net.add(FCLayer(50, 10))
    net.add(ActivationLayer('sigmoid'))

    net.fit(x_train[0:1000], y_train[0:1000], epochs=1000, learning_rate=0.1, silent=False, threshold=0.01, patience=10)
    net.show()
    net.save('mnist.pkl')
    print("--- %s seconds ---" % (time.time() - start_time))
    accuracy = net.evaluate(x_test, y_test)
    print("accuracy : ", accuracy * 100, "%")
    net.disp_loss_accuracy_graph()
    return net

if __name__ == "__main__":
    if os.path.exists('mnist.pkl'):
        net = Network()
        net.load('mnist.pkl')
    else:
        net = save()

    net.summary()
    net.confusion_matrix(x_test, y_test)

    out = net.predict(x_test[0:3])
    for i in range(len(out)):
        img = x_test[i].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.show()
        print("predicted value : ", np.argmax(out[i]), end="\n")
        print("true value : ", np.argmax(y_test[i]), end="\n")
