"""
    The basic building blocks of Neural Network
"""
from typing import Dict, Callable

import numpy as np
from nn.tensor import Tensor
from nn.utils import tanh, tanh_prime, sigmoid, sigmoid_prime, relu, relu_prime


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError


class Dense(Layer):

    """
        Linear Layer computes

        Z = XW + b
    """

    def __init__(self, input_size: int, output_size: int) -> None:

        super().__init__()

        self.params["weights"] = np.random.normal(size=(input_size, output_size))
        self.params["bias"] = np.zeros(output_size)

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        # print(
        #     "Forward",
        #     self.inputs.shape,
        #     self.params["weights"].shape,
        #     self.params["bias"].shape,
        #     (self.inputs @ self.params["weights"] + self.params["bias"]).shape,
        # )
        return self.inputs @ self.params["weights"] + self.params["bias"]

    def backward(self, grad: Tensor) -> Tensor:

        """
            if y = f(z) and z = x*w + b

            dy/dw  = f'(z)*x
            dy/dx = f'(z)*w
            dy/db = f'(z)

            if y = f(z) and z = x@w + b

            dy/dw  = x.T @ f'(z)
            dy/dx = f'(z) @ w.T
            dy/db = f'(z)
        """

        self.grads["bias"] = np.sum(grad, axis=0)
        self.grads["weights"] = self.inputs.T @ grad

        return grad @ self.grads["weights"].T


F = Callable[[Tensor], Tensor]


class Activation(Layer):
    def __init__(self, f: F, f_prime: F) -> None:

        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs

        return self.f(self.inputs)

    def backward(self, grad: Tensor) -> Tensor:

        """
            y = f(x) , x = g(z)

            dy/dz  = f'(x)*g'(z)
        """

        return self.f_prime(self.inputs) * grad

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)


class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)


class Sigmoid(Activation):
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)


class Relu(Activation):
    def __init__(self):
        super().__init__(relu, relu_prime)
