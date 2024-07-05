from node import Node
from mlp import MLP


def naive_example():
    a = Node(1)
    b = Node(-2)
    c = a * b + a  # c.data = -2+1=-1
    d = c * b + b * 0.65  # d.data = 2-1.3=0.7
    d.backward()
    print(f"Gradient of a: {a.gradient}")
    print(f"Gradient of b: {b.gradient}")
    print(f"Gradient of c: {c.gradient}")
    print(f"Gradient of d: {d.gradient}")


def mlp_example():
    x = [1.0, 30.0, -10.0]
    network = MLP(3, [4, 4, 1])
    print(network(x))


def mlp_learning():
    X = [
        [1.0, 3.0, -1.0],
        [3.0, -1.0, 2.5],
        [0.5, 3.0, 1.0],
        [1.5, 1.0, -4.0],
    ]
    y = [1.0, 1.0, -1.0, -1.0]

    n = MLP(3, [4, 4, 1])

    for epoch in range(20):
        y_pred = [n(x) for x in X]
        loss = sum((y_hat - y_true) ** 2 for y_true, y_hat in zip(y, y_pred))
        for p in n.parameters():
            p.gradient = 0.0
        loss.backward()

        # update
        for p in n.parameters():
            p.data += -0.1 * p.gradient

        print(f"Epoch: {epoch}, Loss: {loss.data}")
        
    for i, y_hat in enumerate(y_pred):
        print(f"Predicted value for input {i}: {y_hat.data}")


if __name__ == "__main__":
    mlp_learning()
