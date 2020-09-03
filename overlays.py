from network.nn import NeuralNetwork
import numpy as np


class SGDNeuralNetwork:

    def __init__(self):
        self.prediction = None
        self.X = None
        self.y = None


    def fit(self, X, y, sgd=True, batch=25,
            hidden_layers=1, hidden_neurons=5, 
            l_rate=0.5, rmse_threshold=0.03, random=0):
        self.X = X
        self.y = y
        self.categories = len(set(y))
        
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        self.l_rate = l_rate
        self.rmse_threshold = rmse_threshold
        self.seed = np.random.RandomState(random)
        
        self.nns = []

        if not sgd:
            self.sgd_nn = self.fit_single(self.X, self.y)

        else:
            X_separated = self.create_batches(self.X, batch)
            y_separated = self.create_batches(self.y, batch)

            for curr_X, curr_y in zip(X_separated, y_separated):
                curr_nn = self.fit_single(curr_X, curr_y)
                # print("Expected:", curr_y)
                print("Test Prediction:", curr_nn.forward_prop(curr_X))
                self.nns.append(curr_nn)

            # print(self.nns)
            self._average_weights()
        
        return self


    def fit_single(self, X, y):
        nn, result = NeuralNetwork(X, y, categories=self.categories,
                                    hidden_layers=self.hidden_layers,
                                    hidden_neurons=self.hidden_neurons, 
                                    l_rate=self.l_rate,
                                    random=self.seed,
                                    rmse_threshold=self.rmse_threshold).predict()
        print(f"Result: {result}")

        return nn


    def predict(self, X):
        prediction = self.sgd_nn.forward_prop(X)
        return np.argmax(prediction, axis=1)


    @staticmethod
    def create_batches(lst, n):
        """Separates a `lst` into batches of size `n`."""
        result = []

        for i in range(0, len(lst), n):
            result.append(lst[i:i + n])

        return np.array(result)
    

    def _average_weights(self):
        weights_by_layer = {}
        layers = len(self.nns[0].layers)
        self.sgd_nn = self.nns[0]

        for curr_nn in self.nns:

            for i in range(layers):
                
                if i not in weights_by_layer:
                    weights_by_layer[i] = [curr_nn.layers[i].weights]
                
                else:
                    weights_by_layer[i].append(curr_nn.layers[i].weights)

        for i in weights_by_layer:
            self.sgd_nn.layers[i].weights = np.mean(weights_by_layer[i], axis=0)
