"""
    Data Iterator
"""

from typing import Iterator, NamedTuple

from nn.tensor import Tensor

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])


class DataIterator:
    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        raise NotImplementedError


class DataLoader(DataIterator):
    def __init__(self, batch_size: int = 32) -> None:
        self.batch_size = batch_size

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:

        assert len(inputs) == len(targets), "Input shape and Target shape do not match."
        n = len(inputs)

        for idx in range(0, n, self.batch_size):
            batch_x = inputs[idx : min(idx + self.batch_size, n)]
            batch_y = targets[idx : min(idx + self.batch_size, n)]
            yield Batch(batch_x, batch_y)
