from matplotlib import pyplot as plt
import numpy as np

from network.nn import NeuralNetwork
from template_data import X, y


N_INPUTS = len(X[0])


def visualize_data(X=X, y=y):
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])


    y_pred = NeuralNetwork(X, y).predict()
    # print(y_pred)
