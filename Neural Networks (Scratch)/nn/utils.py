"""
    Utility / Helper functions
"""

import numpy as np
from nn.tensor import Tensor


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)


def tanh_prime(x: Tensor) -> Tensor:

    y = tanh(x)

    return 1 - y ** 2


def sigmoid(x: Tensor) -> Tensor:

    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: Tensor) -> Tensor:

    y = sigmoid(x)

    return y * (1 - y)


# # https://jamesmccaffrey.wordpress.com/2017/06/23/two-ways-to-deal-with-the-derivative-of-the-relu-function/


# def relu(x: Tensor) -> Tensor:
#     return np.log(1 + np.exp(x))


# def relu_prime(x: Tensor) -> Tensor:
#     return sigmoid(x)


def relu(x: Tensor) -> Tensor:
    return max(0, x)


def relu_prime(x: Tensor) -> Tensor:
    x[x <= 0] = 0
    x[x < 0] = 1

    return x
