from node import Node
import random
from typing import Union


class Neuron:
    def __init__(self, n_input: int) -> None:
        self.w = [Node(random.uniform(-1, 1)) for _ in range(n_input)]
        self.b = Node(random.uniform(-1, 1))

    def __call__(self, x: list[Node]) -> Node:
        # w * x + b
        act = sum((w_i * x_i for w_i, x_i in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self) -> list[Node]:
        return self.w + [self.b]


class Layer:
    def __init__(self, n_input: int, n_output: int) -> None:
        self.neurons = [Neuron(n_input) for _ in range(n_output)]

    def __call__(self, x: Neuron) -> Union[Neuron, list[Neuron]]:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> list[Neuron]:
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    def __init__(self, n_input: int, n_output) -> None:
        sz = [n_input] + n_output
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(n_output))]

    def __call__(self, x: Layer):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Layer]:
        return [p for layer in self.layers for p in layer.parameters()]
