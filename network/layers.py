import numpy as np


class Layer:
    """
    Layer object, with it's list of weights and biases

    Args:
        n_inputs (int): Number of layer inputs (the neurons of the previous layer)
        n_neurons (int): Number of layer neurons
    """

    def __init__(self, n_inputs, n_neurons, random=None):
        self.weights = random.randn(n_inputs, n_neurons)
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
        # print("Inputs Shape: ", inputs.shape)
        # print("Weights Shape: ", self.weights.shape)
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        # print("Outputs Shape: ", self.output.shape)
        # print("----")



    def activation_derivative(self):
        # In this case we will only implement for a Sigmoid function
        return self.output * (1 - self.output)
        
