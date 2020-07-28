from matplotlib import pyplot as plt

from nn import NeuralNetwork
from template_data import X, y


N_INPUTS = len(X[0])


def visualize_data(X=X, y=y):
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


if __name__ == "__main__":
    y_pred = NeuralNetwork(X, y).forward_prop()
    # print(y_pred)
    print("\n\n")
    # print(y)