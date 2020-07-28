import numpy as np


class Layer:
    """
    Layer object, with it's list of weights and biases

    Args:
        n_inputs (int): Number of layer inputs (the neurons of the previous layer)
        n_neurons (int): Number of layer neurons
    """

    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        # print(f"WEIGHTS: {self.weights}\n\n")
        self.biases = np.zeros((1, n_neurons))


    def activation(activation_type):

        def inner(func):

            def wrapper(self, inputs):
                func(self, inputs)

                if activation_type == "relu":
                    self.output = np.maximum(0, self.output)

                elif activation_type == "sigm":
                    sigmoid = lambda x: 1/(1 + np.exp(-x))
                    sigm = np.vectorize(sigmoid)
                    self.output = sigm(self.output)

                else:
                    print("There is no such Activation Function")

                return self.output 
                
            return wrapper

        return inner


    @activation("sigm")
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases



