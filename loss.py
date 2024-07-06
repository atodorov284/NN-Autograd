from node import Node
from abc import ABC, abstractmethod


class Loss(ABC):
    def __init__(self) -> None:
        self._internal_loss = Node(data=0.0)

    @abstractmethod
    def __call__(self, prediction: list, target: list) -> float:
        pass

    @property
    def data(self) -> float:
        return self._internal_loss.data

    def backward(self) -> None:
        self._internal_loss.backward()


class SquaredLoss(Loss):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, prediction: list, target: list) -> float:
        self._internal_loss = sum(
            (y_hat - y_true) ** 2 for y_true, y_hat in zip(target, prediction)
        )
        return self.data
