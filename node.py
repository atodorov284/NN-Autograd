from __future__ import annotations
import math


class Node:
    """
    A data structure class to be able to process nodes that have a stored value and a gradient
    """

    def __init__(
        self, data: int, _children: tuple[Node, Node] = (), _operation: str = ""
    ) -> None:
        self.data = data
        self.gradient = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._operation = _operation

    def __repr__(self):
        return f"Node(data={self.data}, gradient={self.gradient})"

    def __add__(self, other_node: Node) -> Node:
        # If a scalar is given cast it to Node
        if not isinstance(other_node, Node):
            other_node = Node(other_node)

        out = Node(self.data + other_node.data, (self, other_node), "+")

        def _backward():
            # Gradient rule for addition
            self.gradient += 1.0 * out.gradient
            other_node.gradient += 1.0 * out.gradient

        out._backward = _backward

        return out

    def __radd__(self, other_node: Node) -> Node:
        return self + other_node

    def __mul__(self, other_node: Node) -> Node:
        if not isinstance(other_node, Node):
            other_node = Node(other_node)

        out = Node(self.data * other_node.data, (self, other_node), "*")

        def _backward():
            # Gradient rule for multiplication
            self.gradient += other_node.data * out.gradient
            other_node.gradient += self.data * out.gradient

        out._backward = _backward

        return out

    def __rmul__(self, other_node: Node) -> Node:
        return self * other_node

    def __neg__(self) -> Node:
        return self * -1

    def __sub__(self, other_node: Node) -> Node:
        return self + (-other_node)

    def __pow__(self, other_node: Node) -> Node:
        out = Node(self.data**other_node, (self,), f"**{other_node}")

        def _backward():
            self.gradient += other_node * (self.data ** (other_node - 1)) * out.gradient

        out._backward = _backward

        return out

    def tanh(self) -> Node:
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Node(t, (self,), "tanh")

        def _backward():
            self.gradient += (1 - t**2) * out.gradient

        out._backward = _backward

        return out
    
    def backward(self) -> None:
        # Topological sort to visit each node once and compute its gradient
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.gradient = 1.0
        for node in reversed(topo):
            node._backward()
