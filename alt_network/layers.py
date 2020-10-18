import numpy as np


class Layer:

    def __init__(self, n_inputs, n_neurons):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Will be calculated in the optimizers' module - optimizers.py
        self.errors = None


    def forward(self, input_):
        self.output = input_.dot(self.weights) + self.biases
        self.output = self.sigmoid_activation(self.output)
        
        return self.output


    @staticmethod
    def sigmoid_activation(neurons):
        sigmoid = lambda x: 1/(1 + np.exp(-x))
        sigm = np.vectorize(sigmoid)
        neurons = sigm(neurons)

        return neurons


    @staticmethod
    def sigmoid_derivative(activation_outputs):
        return activation_outputs * (1 - activation_outputs)


    def create_deltas(self):
        self.deltas = self.errors * self.sigmoid_derivative(self.output)