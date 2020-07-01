"""
    The example of xor which cannot be learned by a linear model
"""

from nn.model import Sequential
from nn.layers import Dense, Tanh, Sigmoid, Relu
from nn.train import train
import numpy as np

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0.0], [1.0], [1.0], [0.0]])

net = Sequential()
net.add(Dense(2, 2))
net.add(Tanh())
net.add(Dense(2, 1))
net.add(Sigmoid())

train(net, inputs, targets)
print(net(np.expand_dims(inputs[0], 0)))
print(inputs[0].shape)

for X, y in zip(inputs, targets):
    predicted = net(np.expand_dims(X, 0))

    print(X, predicted, y)

