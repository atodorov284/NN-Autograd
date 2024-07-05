from abc import ABC, abstractmethod
from node import Node


class Optimizer(ABC):
    def __init__(self, parameters: list[Node], learning_rate: float = 0.01) -> None:
        self._parameters = parameters
        self._learning_rate = learning_rate

    @abstractmethod
    def step(self) -> None:
        pass


class SGD(Optimizer):
    def __init__(self, parameters: list[Node], learning_rate: float = 0.01) -> None:
        super().__init__(parameters, learning_rate)

    def step(self) -> None:
        for p in self._parameters:
            p.data += (-self._learning_rate) * p.gradient
