from matplotlib import pyplot as plt
import numpy as np

# from network.nn import NeuralNetwork
from overlays import SGDNeuralNetwork
from template_data import X, y


def visualize_data(X=X, y=y):
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


def shuffle_data(X, y, state=0):
    idx = np.random.RandomState(state).permutation(len(X))
    X, y = X[idx], y[idx]


def preprocess(X, y):
    shuffle_data(X, y)
    data_size = len(X)
    train_size = int(data_size * 0.70)

    return X[:train_size], y[:train_size], X[train_size:], y[train_size:]


def error(y, y_pred):
    differences = 0

    for y1, y2 in zip(y, y_pred):
        if y1 != y2:
            differences += 1

    return differences / len(y)


if __name__ == "__main__":
    # # Simple boolean example:
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y = np.array([0, 0, 0, 1])


    X_train, y_train, X_test, y_test = preprocess(X, y)
    # X_train = np.array([[ 0.16486876, 0.18793162], [ 0.41615746, -0.27715153]])
    # y_train = np.array([1, 1])

    model = SGDNeuralNetwork().fit(X_train, y_train,
                                   hidden_layers=4, 
                                   hidden_neurons=4,
                                   l_rate=0.5,
                                   random=0,
                                   rmse_threshold=0.25)
    # y_pred = model.predict(X_train)
    # print("Data Size:", len(X))
    # print("Expected :", y_train)
    # print("Predicted:", y_pred)
    # print(f"Error:", error(y, y_pred))

