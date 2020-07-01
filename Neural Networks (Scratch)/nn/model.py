"""
    The structure of the neural network
"""

from nn.tensor import Tensor
from nn.layers import Layer

from typing import List, Iterator, Tuple


class Model:
    def __call__(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, X: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError

    def predict(self, X: Tensor) -> Tensor:
        raise NotImplementedError

    def get_params_grad(self) -> Iterator[Tuple[Tensor, Tensor]]:
        raise NotImplementedError


class Sequential(Model):
    def __init__(self, layers: List[Layer] = []) -> None:
        self.layers = layers

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def forward(self, X: Tensor) -> Tensor:

        for layer in self.layers:
            X = layer(X)
        return X

    def backward(self, grad: Tensor) -> Tensor:

        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def get_params_grad(self) -> Iterator[Tuple[Tensor, Tensor]]:

        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]

                yield param, grad
