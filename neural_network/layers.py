import numpy as np

from neural_network.activations import NeuronActivations, ACTIVATIONS

class Layer(NeuronActivations):

    def __init__(self, n_inputs, n_neurons, random_state, activation):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation = activation

        np.random.seed(random_state)
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Will be calculated in the optimizers' module - optimizers.py
        self.errors = None

    def forward(self, input_):
        self.output = input_.dot(self.weights) + self.biases
        self.output = eval(f"self.{ACTIVATIONS[self.activation]}_activation(self.output)")
        
        return self.output

    def create_deltas(self):
        self.deltas = self.errors * eval(
            f"self.{ACTIVATIONS[self.activation]}_derivative(self.output)"
        )