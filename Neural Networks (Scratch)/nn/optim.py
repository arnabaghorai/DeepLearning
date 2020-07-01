"""
    Optimizer adjusts and updates the parameters of our neural network
"""
from nn.model import Model


class Optimizer:
    def step(self, net: Model) -> None:
        raise NotImplementedError


class SGD(Optimizer):

    """
        Stocastic Gradient Descent
    """

    def __init__(self, lr: float = 0.0001) -> None:
        self.lr = lr

    def step(self, net: Model) -> None:

        for param, grad in net.get_params_grad():
            # print(param.shape, grad.shape)
            param -= self.lr * grad
