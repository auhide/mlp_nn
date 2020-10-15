import numpy as np

from alt_network.layers import Layer
from alt_network.exceptions import WrongLayerFormat



class NeuralNetwork:

    def __init__(self, X, y, learning_rate=0.5, epochs=100):
        self.X = X
        self.y = y

        self.input_neurons = len(X[0])
        self.output_neurons = len(set(y))
        self.l_rate = learning_rate
        self.epochs = epochs

        self._layers = []


    def fit(self):
        self._backpropagation()


    def predict(self):
        prediction = self._convert_as_prediction(self._layers[-1].output)
        
        return np.array(prediction)


    @staticmethod
    def _convert_as_prediction(raw_prediction):
        result = []
        print(raw_prediction)

        for row in raw_prediction:
            result.append(list(row).index(max(row)))

        return result


    def display_layers(self):
        print(f"Layer[0]: (1, {self.input_neurons})")

        for i in range(len(self._layers)):
            print(f"Layer[{i+1}]: ({self._layers[i].n_inputs}, {self._layers[i].n_neurons})")


    def _forward(self):

        if not self._output_layer_is_valid():
            raise WrongLayerFormat(f"Last layer's neurons have to be {self.output_neurons}")

        for i, layer in enumerate(self._layers):
            
            if i == 0:
                prev_output = layer.forward(self.X)
            
            else:
                prev_output = layer.forward(prev_output)

        self.output = self._layers[-1].output


    def _backpropagation(self):
        self._forward()
        self._reformat_output()

        for _ in range(self.epochs):
            self._gradient_descent()


    def add_layer(self, neurons):
        
        # When adding the first layer
        if not len(self._layers):
            self._layers.append(
                Layer(self.input_neurons, neurons)
            )
        
        # Adding all other layers
        else:
            prev_layer_inputs = self._layers[len(self._layers)-1].n_neurons
            self._layers.append(
                Layer(prev_layer_inputs, neurons)
            )

    
    def _update_weights(self):
        
        for i in range(len(self._layers)):
            curr_input = np.atleast_2d(self.X if i == 0 else self._layers[i-1].output)
            weights_change = self._layers[i].deltas.T.dot(curr_input) * self.l_rate

            self._layers[i].weights += weights_change.T


    def _gradient_descent(self):

        for i in reversed(range(len(self._layers))):
            
            if self._layers[i] == self._layers[-1]:
                self._layers[i].errors = self.expected_output - self.output
                self._layers[i].create_deltas()

            else:
                next_layer = self._layers[i+1]
                self._layers[i].errors = next_layer.deltas.dot(next_layer.weights.T)
                self._layers[i].create_deltas()

        self._update_weights()
        self._forward()


    def _output_layer_is_valid(self):
        """True if the expected output layer's neurons are equal to the latest \
        added layer's neurons, else - False

        Returns:
            bool: Whether the out layer is valid or not
        """
        return self._layers[len(self._layers)-1].n_neurons == self.output_neurons

    
    def _reformat_output(self):
        """
        Converts the vector of expected results
        to a matrix, based on the neurons of the last layer.

        E.g. the neurons of the output layer are 3, the 
        expected results' vector will be converted to mx3,
        where m is the number of the expected results.
        """
        expected_converted = np.zeros((
            len(self.y),
            self.output_neurons
        ))
        # print(f"Expected: {self.y}")

        for i, output in enumerate(self.y):
            # print(f"i={i} ; output={output}")
            expected_converted[i][output] = 1

        self.expected_output = expected_converted
        print(f"Changed Expected: {self.expected_output}")



if __name__ == "__main__":
    pass