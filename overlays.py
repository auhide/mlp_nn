from network.nn import NeuralNetwork
import numpy as np


class SGDNeuralNetwork:

    def __init__(self):
        self.prediction = None
        self.X = None
        self.y = None


    def fit(self, X, y, 
            hidden_layers=1, hidden_neurons=5, 
            l_rate=0.5, rmse_threshold=0.03, random=0):
        self.X = X
        self.y = y
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.l_rate = l_rate
        self.rmse_threshold = rmse_threshold
        self.seed = np.random.RandomState(random)
        self.nns = []

        X_separated = self.split_by_two(self.X)
        y_separated = self.split_by_two(self.y)
        print(y_separated)
        counter = 0

        for curr_X, curr_y in zip(X_separated, y_separated):
            print(counter)
            counter += 1
            print(curr_X)
            print(curr_y)
            curr_nn = self.fit_single(curr_X, curr_y)
            self.nns.append(curr_nn)

        print(self.nns)
        
        return self


    def fit_single(self, X, y):
        nn, result = NeuralNetwork(X, y, 
                                    hidden_layers=self.hidden_layers,
                                    hidden_neurons=self.hidden_neurons, 
                                    l_rate=self.l_rate,
                                    random=self.seed,
                                    rmse_threshold=self.rmse_threshold).predict()

        return nn


    def predict(self, X):
        prediction = self.nn.forward_prop(X)
        return np.argmax(prediction, axis=1)


    @staticmethod
    def split_by_two(lst):

        if len(lst) == 1:
            return np.array([lst])

        if len(lst) % 2 == 0:
                result = [lst[i-1:i+1] for i in range(1, len(lst), 2)]

        else:
                result = [lst[i-1:i+1] for i in range(1, len(lst), 2)] + [[lst[-1]]]

        return np.array(result)
    