import numpy as np


ACTIVATIONS = {
    "sigm": "sigmoid",
    "tanh": "tanh",
    "relu": "rectifier",
}


class BaseActivation:

    @classmethod
    def activate(cls, activation, neurons):
        func = np.vectorize(activation)
        neurons = func(neurons)

        return neurons


class ReLU(BaseActivation):

    @classmethod
    def rectifier_activation(cls, neurons):
        activation = lambda x: x if x > 0 else 0

        return cls.activate(activation, neurons)

    @classmethod
    def rectifier_derivative(cls, activation_outputs):
        derivative = lambda x: 1 if x > 0 else 0

        return cls.activate(derivative, activation_outputs)



class Sigmoid(BaseActivation):

    @classmethod
    def sigmoid_activation(cls, neurons):
        activation = lambda x: 1/(1 + np.exp(-x))

        return cls.activate(activation, neurons)

    @classmethod
    def sigmoid_derivative(cls, activation_outputs):
        return activation_outputs * (1 - activation_outputs)


class TanH(BaseActivation):

    @classmethod
    def tanh_activation(cls, neurons):
        activation = lambda x: (np.exp(2 * x) - 1 ) / (np.exp(2 * x) + 1)

        return cls.activate(activation, neurons)

    @classmethod
    def tanh_derivative(cls, activation_outputs):
        return 1 - (activation_outputs ** 2)


class NeuronActivations(Sigmoid, TanH, ReLU):
    pass
