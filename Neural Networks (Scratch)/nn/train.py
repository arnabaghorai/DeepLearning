"""
    Train the model
"""

from nn.tensor import Tensor
from nn.model import Model
from nn.loss import Loss, MSE
from nn.data import DataIterator, DataLoader
from nn.optim import Optimizer, SGD


def train(
    net: Model,
    inputs: Tensor,
    targets: Tensor,
    epochs: int = 100000,
    iterator: DataIterator = DataLoader(),
    loss: Loss = MSE(),
    optim: Optimizer = SGD(),
) -> None:

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in iterator(inputs, targets):
            predicted = net(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)

            net.backward(grad)
            optim.step(net)

            print(f"Epoch: {epoch} --> Loss: {epoch_loss}")

