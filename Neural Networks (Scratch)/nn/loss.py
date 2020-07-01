"""
    Loss helps use to evaluate our predictions
"""

from nn.tensor import Tensor
import numpy as np


class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:

        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):

    """

        Calculates the Mean Squared Error Loss

        L = ((y_hat - y)**2)/2m , where m = no of samples
    """

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return 0.5 * np.mean((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        # print("loss:", (predicted - actual).shape)
        return predicted - actual
