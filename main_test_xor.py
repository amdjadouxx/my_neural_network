import numpy as np
import os
import time
from neural_network.Network import Network
from neural_network.FCLayer import FCLayer
from neural_network.ActivationLayer import ActivationLayer

x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

def load():
    net = Network()
    net.load('xor.pkl')
    return net

def save():
    start_time = time.time()
    net = Network('mse')
    net.add(FCLayer(2, 3))
    net.add(ActivationLayer('tanh'))
    net.add(FCLayer(3, 1))
    net.add(ActivationLayer('sigmoid'))

    net.fit(x_train, y_train, epochs=500, learning_rate=0.1, silent=True, threshold=0.01, patience=10)
    net.disp_loss_accuracy_graph()
    net.save('xor.pkl')
    print("--- %s seconds ---" % (time.time() - start_time))
    return net

def prediction(net):
    out = net.predict(x_train)
    for vidx in range(len(out)):
        if out[vidx] > 0.5:
            out[vidx] = 1
        else:
            out[vidx] = 0
    print(out)

if __name__ == '__main__':
    if os.path.exists('xor.pkl'):
        net = load()
    else:
        net = save()
    net.summary()
    net.show()

    prediction(net)