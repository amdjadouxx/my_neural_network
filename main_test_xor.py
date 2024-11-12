import numpy as np
import os
from FCLayer import FCLayer
from ActivationLayer import ActivationLayer
from ActivationFunc import *
from LossesFunc import *
from Network import Network
from DropoutLayer import DropoutLayer

x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

def load():
    net = Network()
    net.load('xor.pkl')
    return net

def save():
    net = Network(loss=mse, loss_prime=mse_prime)
    net.add(FCLayer(2, 3))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(3, 1))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))

    net.fit(x_train, y_train, epochs=2000, learning_rate=0.1, silent=True)
    net.save('xor.pkl')
    return net


if __name__ == '__main__':
    if os.path.exists('xor.pkl'):
        net = load()
    else:
        net = save()
    net.summary()
    net.disp_loss_accuracy_graph()
    net.show()

    out = net.predict(x_train)
    for vidx in range(len(out)):
        if out[vidx] > 0.5:
            out[vidx] = 1
        else:
            out[vidx] = 0
    print(out)