import numpy as np

from neural_network.template_data import X, y
from overlays import NeuralNetFactory
from neural_network.evaluations import Evaluator


def shuffle_data(X, y):
    np.random.seed(1)
    ids_shuffled = np.random.permutation(len(y))
    X = X[ids_shuffled]
    y = y[ids_shuffled]

    return X, y


def preprocess(X, y):
    X, y = shuffle_data(X, y)
    X = normalize_data(X)

    return train_test_split(X, y)


def train_test_split(X, y, train_size=0.7):
    train_size = int(train_size * len(X))
    X_train = X[:train_size]
    X_test = X[train_size:]
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



if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess(X, y)
    X_train = X_train.astype(np.float128)
    
    print(X_train)
    print("\n")
    print(y_train)
    print("-"*100)

    architecture = {
        0: 3,
        1: 3,
        2: 3,
    }

    nn = NeuralNetFactory.define_nn(
        optimizer="sgdm",
        architecture_dict=architecture,
        X=X_train, 
        y=y_train,
        learning_rate=0.5, 
        epochs=10, 
        random=0, 
        activation="sigm"
    )

    nn.fit()
    print("Training Has Finished\n")
    prediction = nn.predict(X_test)
    print("Prediction:\n", prediction)
    print("Expected:\n", y_test)

    accuracy = Evaluator.accuracy(y_test, prediction)
    print("\nAccuracy: ", accuracy)