import numpy as np

from nn.neural_network.template_data import X, y
from nn.overlays import NeuralNetFactory
from nn.neural_network.evaluations import Evaluator
from preprocess.base import *



def convert_as_prediction(raw_prediction):
        result = []
        # print(raw_prediction)

        for row in raw_prediction:
            result.append(list(row).index(max(row)))

        return np.array(result)


def test():
    X_train, X_test, y_train, y_test = preprocess(X, y)
    X_train = X_train.astype(np.float128)
    
    print(X_train)
    print("\n")
    print(y_train)
    print("-"*100)

    architecture = {
        1: 3,
        2: 3,
        3: 3
    }

    nn = NeuralNetFactory.define_nn(
        optimizer="adam",
        # type_="regression",
        architecture_dict=architecture,
        X=X_train, 
        y=y_train,
        learning_rate=0.1, 
        epochs=5, 
        random=0, 
        activation="sigm",
        epsilon=1e-7
    )

    print(nn.get_architecture())
    nn.fit()
    print("Training Has Finished\n")
    print(nn.predict(X_test))
    prediction = convert_as_prediction(nn.predict(X_test))
    print("Prediction:\n", prediction)
    print("Expected:\n", y_test)

    accuracy = Evaluator.accuracy(y_test, prediction)
    print("\nAccuracy: ", accuracy)


if __name__ == "__main__":
    test()