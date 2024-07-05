import numpy as np
from typing import Union

def train_validation_test_split(X: np.array, y: np.array, train_val_ratio: int = 0.8, train_test_ratio: int = 0.8, shuffle=True) -> Union[np.array, np.array, np.array]:
    assert len(X) == len(y), "X and y must have the same length"
    
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
    
    train_size = int(len(X) * train_test_ratio)
    test_size = len(X) - train_size
    train_size = int(train_size * train_val_ratio)
    val_size = len(X) - train_size - test_size
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test