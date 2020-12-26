import numpy as np

from nn.neural_network.layers import Layer
from nn.neural_network.exceptions import WrongLayerFormat



class NeuralNetwork:

    def __init__(self, X, y, 
                 learning_rate=0.1, 
                 epochs=10, 
                 batch=5,
                 random=0,
                 activation="sigm",
                 type=False,
                 momentum=0.5,
                 epsilon=1e-7,
                 beta_1=0.9,
                 beta_2=0.999):
        self.X = X
        self.y = y

        self.batch_size = batch
        self.input_neurons = len(X[0])
        self.output_neurons = len(set(y))
        self.l_rate = learning_rate
        self.epochs = epochs
        self.random = random
        self.activation = activation
        self.momentum = momentum
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self._layers = []

    def fit(self):
        self._backpropagation()

    def predict(self, X):
        self._forward(X)

        if self._layers[-1].n_neurons > 1:
            prediction = self._layers[-1].output
            prediction = np.array(prediction)

        else:
            prediction = self._layers[-1].output
            prediction = prediction.flatten()

        return prediction

    def get_architecture(self):
        """Creates a dictionary describing the architecture of the NN

        Returns:
            dict: Architecture of the NN
        """

        architecture = {}
        architecture[0] = self.input_neurons

        for i in range(len(self._layers)):
            architecture[i+1] = self._layers[i].n_neurons

        return architecture

    def add_layer(self, neurons):
        """Adds new layer to your NeuralNet object

        Args:
            neurons (int): number of neurons of the added layer
        """
        
        # When adding the first layer
        if not len(self._layers):
            self._layers.append(
                Layer(self.input_neurons, neurons, self.random, self.activation)
            )
        
        # Adding all other layers
        else:
            prev_layer_inputs = self._layers[len(self._layers)-1].n_neurons
            self._layers.append(
                Layer(prev_layer_inputs, neurons, self.random, self.activation)
            )

    def _forward(self, X):

        if not self._output_layer_is_valid():
            raise WrongLayerFormat(
                f"Last layer's neurons have to be {self.output_neurons}"
            )

        for i, layer in enumerate(self._layers):
            
            if i == 0:
                prev_output = layer.forward(X)
            
            else:
                prev_output = layer.forward(prev_output)

        self.output = self._layers[-1].output

        return self.output

    def _update_weights(self, X):
        
        for i in range(len(self._layers)):
            curr_input = np.atleast_2d(
                X if i == 0 else self._layers[i-1].output
            )
            weights_change = self._layers[i].deltas.T.dot(curr_input) * self.l_rate

            self._layers[i].weights += weights_change.T

    def _output_layer_is_valid(self):
        """True if the expected output layer's neurons are equal to the latest \
        added layer's neurons, else - False

        Returns:
            bool: Whether the out layer is valid or not
        """
        return self._layers[-1].n_neurons == self.output_neurons

    def _reformat_output(self, y):
        """
        Converts the vector of expected results
        to a matrix, based on the neurons of the last layer.

        E.g. the neurons of the output layer are 3, the 
        expected results' vector will be converted to mx3,
        where m is the number of the expected results.
        """
        expected_converted = np.zeros((
            len(y),
            self.output_neurons
        ))

        for i, output in enumerate(y):
            expected_converted[i][output] = 1

        self.expected_output = expected_converted
