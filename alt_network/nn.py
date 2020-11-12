import numpy as np

from alt_network.layers import Layer
from alt_network.exceptions import WrongLayerFormat



class NeuralNetwork:

    def __init__(self, X, y, 
                 learning_rate=0.5, 
                 epochs=50, 
                 batch=5,
                 random=0,
                 activation="sigm"):
        self.X = X
        self.y = y

        self.batch_size = batch
        self.input_neurons = len(X[0])
        self.output_neurons = len(set(y))
        self.l_rate = learning_rate
        self.epochs = epochs
        self.random = random
        self.activation = activation

        self._layers = []


    def fit(self):
        self._backpropagation()


    def predict(self, X):
        self._forward(X)
        prediction = self._convert_as_prediction(self._layers[-1].output)
        
        return np.array(prediction)


    def display_layers(self):
        print(f"Layer[0]: (1, {self.input_neurons})")

        for i in range(len(self._layers)):
            print(f"Layer[{i+1}]: ({self._layers[i].n_inputs}, {self._layers[i].n_neurons})")


    def add_layer(self, neurons):
        
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


    @staticmethod
    def _convert_as_prediction(raw_prediction):
        result = []
        # print(raw_prediction)

        for row in raw_prediction:
            result.append(list(row).index(max(row)))

        return result

    
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
