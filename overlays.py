from network.nn import NeuralNetwork
import numpy as np


class SGDNeuralNetwork:

    def __init__(self):
        self.prediction = None

    
    def fit(self, X, y, 
            hidden_layers=1, hidden_neurons=5, 
            l_rate=0.5, rmse_threshold=0.03, random=0):
        self.X = X
        self.y = y
        self.seed = np.random.RandomState(random)

        self.nn, self.result = NeuralNetwork(self.X, self.y, 
                               hidden_layers=hidden_layers,
                               hidden_neurons=hidden_neurons, 
                               l_rate=l_rate,
                               random=self.seed,
                               rmse_threshold=rmse_threshold).predict()

        return self

    
    def predict(self, X):
        prediction = self.nn.forward_prop(X)
        return np.argmax(prediction, axis=1)
    