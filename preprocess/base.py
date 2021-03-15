import numpy as np
from config import DATASET_SIZE


def shuffle_data(X, y):
    np.random.seed(1)
    ids_shuffled = np.random.permutation(len(y))
    X = X[ids_shuffled]
    y = y[ids_shuffled]

    return X, y


def preprocess(X, y):
    X, y = shuffle_data(X, y)
    X = normalize_data(X)
    X = X[:DATASET_SIZE]
    y = y[:DATASET_SIZE]

    return train_test_split(X, y)


def train_test_split(X, y, train_size=0.7):
    train_size = int(train_size * len(X))
    X_train = X[:train_size].astype(np.float128)
    X_test = X[train_size:].astype(np.float128)
    y_train = y[:train_size]
    y_test = y[train_size:]

    return X_train, X_test, y_train, y_test


def normalize_data(X):
    cols_max = X.max(axis=0)
    X_transposed = X.T

    for col_i in range(len(cols_max)):
        X_transposed[col_i] = X_transposed[col_i] / cols_max[col_i]
    
    X = X_transposed.T

    return X