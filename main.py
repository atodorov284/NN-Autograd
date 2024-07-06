from node import Node
from mlp import MLP
from optimizer import SGD
from batch_loader import BatchLoader
import numpy as np
from utils import train_validation_test_split
from loss import SquaredLoss


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
    classes = 2
    size = 1000
    noise = 0.1

    y = np.random.randint(0, classes, size)
    class_1 = (np.random.rand(size) + y) / classes
    class_2 = class_1 + np.random.rand(size) * noise
    X = np.stack([class_1, class_2], axis=1)

    network = MLP(2, [4, 4, 1])

    optimizer = SGD(network.parameters(), 0.01)
    X_train, y_train, X_val, y_val, X_test, y_test = train_validation_test_split(X, y)

    loader = BatchLoader(X_train, y_train, 10)
    loss = SquaredLoss()
    for epoch in range(20):
        for batch_x, batch_y in loader:
            y_pred = [network(x) for x in batch_x]
            # loss = sum((y_hat - y_true) ** 2 for y_true, y_hat in zip(batch_y, y_pred))
            loss(y_pred, batch_y)
            network.zero_grad()

            loss.backward()

            optimizer.step()
        print(f"Epoch: {epoch}, Training Loss: {loss.data}")

        y_val_pred = [network(x) for x in X_val]
        val_loss = loss(y_val_pred, y_val)
        print(f"Epoch: {epoch}, Validation Loss: {val_loss}")

    y_test_pred = [network(x) for x in X_test]
    test_loss = loss(y_test_pred, y_test)
    print(f"Final Test Loss: {test_loss}")


if __name__ == "__main__":
    mlp_learning()
