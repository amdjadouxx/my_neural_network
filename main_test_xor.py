import numpy as np
from FCLayer import FCLayer
from ActivationLayer import ActivationLayer
from ActivationFunc import *
from LossesFunc import *
from Network import Network

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(sigmoid, sigmoid_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=2000, learning_rate=0.1)
net.summary()

# test
out = net.predict(x_train)
for vidx in range(len(out)):
    if out[vidx] > 0.5:
        out[vidx] = 1
    else:
        out[vidx] = 0
print(out)