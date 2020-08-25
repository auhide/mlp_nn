from network.nn import NeuralNetwork


class FFNeuralNetwork:

    def __init__(self):
        self.prediction = None

    
    def train(self, X, y, 
              hidden_layers=1, hidden_neurons=5, 
              l_rate=0.5, rmse_threshold=0.03, random=0):
        
        self.prediction = NeuralNetwork(X, y, 
                          hidden_layers=hidden_layers,
                          hidden_neurons=hidden_neurons, 
                          l_rate=l_rate,
                          random=random,
                          rmse_threshold=rmse_threshold).predict()

        return self

    def predict(self):
        return self.prediction




    