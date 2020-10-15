import numpy as np

from alt_network.template_data import X, y
from alt_network.nn import NeuralNetwork


def shuffle_data(X, y):
    ids_shuffled = np.random.permutation(len(y))
    X = X[ids_shuffled]
    y = y[ids_shuffled]

    return X, y


def preprocess(X, y):
    X, y = shuffle_data(X, y)


if __name__ == "__main__":
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y = np.array([0, 1, 1, 0])

    X, y = shuffle_data(X, y)
    X, y = X[:10], y[:10]
    X = X.astype(np.float128)
    
    print(X)
    print("\n")
    print(y)
    print("-"*100)

    nn = NeuralNetwork(X, y, epochs=3000)
    
    nn.add_layer(neurons=3)
    nn.add_layer(neurons=3)
    nn.add_layer(neurons=3)
    nn.display_layers()
    nn.fit()
    prediction = nn.predict()
    print(prediction)