
import numpy as np
from template_data import X, y


class Layer:

    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))


    def forward(self, input_):
        self.output = input_.dot(self.weights) + self.biases
        self.output = self.sigmoid_activation(self.output)
        print(self.output.shape)


    def sigmoid_activation(self, neurons):
        sigmoid = lambda x: 1/(1 + np.exp(-x))
        sigm = np.vectorize(sigmoid)
        neurons = sigm(neurons)

        return neurons



class NeuralNetwork:

    def __init__(self, X, y, hidden=1, hidden_neurons=3):
        self.X = X
        self.y = y

        self.input_neurons = len(X[0])
        self.output_neurons = len(set(y))
        self.hidden_layers = hidden
        self.hidden_neurons = hidden_neurons


    def forward(self):
        print(f"Input Neurons: {self.input_neurons}")
        print(f"Output Neurons {self.output_neurons}")
        self.layers = [
            Layer(self.input_neurons, self.output_neurons)
        ]
        
        for i in range(1, self.hidden_layers):
            self.layers.append(Layer(self.hidden_neurons, self.hidden_neurons))


if __name__ == "__main__":
    print(X)
    print("\n")
    print(y)
    print("-"*100)

    nn = NeuralNetwork(X, y)
    nn.forward()