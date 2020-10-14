import numpy as np

from template_data import X, y
from layers import Layer
from exceptions import WrongLayerFormat



class NeuralNetwork:

    def __init__(self, X, y, learning_rate=0.1):
        self.X = X
        self.y = y

        self.input_neurons = len(X[0])
        self.output_neurons = len(set(y))
        self.l_rate = learning_rate

        self.layers = []


    def forward(self):

        if not self._output_layer_is_valid():
            raise WrongLayerFormat(f"Last layer's neurons have to be {self.output_neurons}")

        for i, layer in enumerate(self.layers):
            
            if i == 0:
                prev_output = layer.forward(X)
            
            else:
                prev_output = layer.forward(prev_output)

        self.output = self.layers[-1].output


    def backpropagation(self):
        self.forward()
        self._reformat_output()
        
        for _ in range(1000):
            self._gradient_descent()

        print(self.layers[-1].output)
        


    def display_layers(self):
        print(f"Layer[0]: (1, {self.input_neurons})")

        for i in range(len(self.layers)):
            print(f"Layer[{i+1}]: ({self.layers[i].n_inputs}, {self.layers[i].n_neurons})")


    def add_layer(self, neurons):
        
        # When adding the first layer
        if not len(self.layers):
            self.layers.append(
                Layer(self.input_neurons, neurons)
            )
        
        # Adding all other layers
        else:
            prev_layer_inputs = self.layers[len(self.layers)-1].n_neurons
            self.layers.append(
                Layer(prev_layer_inputs, neurons)
            )

    
    def _update_weights(self):
        
        for i in range(len(self.layers)):
            curr_input = np.atleast_2d(self.X if i == 0 else self.layers[i-1].output)
            weights_change = self.layers[i].deltas.T.dot(curr_input) * self.l_rate

            self.layers[i].weights += weights_change.T


    def _gradient_descent(self):

        for i in reversed(range(len(self.layers))):
            
            if self.layers[i] == self.layers[-1]:
                self.layers[i].errors = self.expected_output - self.output
                self.layers[i].create_deltas()

            else:
                next_layer = self.layers[i+1]
                self.layers[i].errors = next_layer.deltas.dot(next_layer.weights.T)
                self.layers[i].create_deltas()

        self._update_weights()
        self.forward()


    def _output_layer_is_valid(self):
        """True if the expected output layer's neurons are equal to the latest \
        added layer's neurons, else - False

        Returns:
            bool: Whether the out layer is valid or not
        """
        return self.layers[len(self.layers)-1].n_neurons == self.output_neurons

    
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
        print(f"Expected: {self.y}")

        for i, output in enumerate(self.y):
            print(f"i={i} ; output={output}")
            expected_converted[i][output] = 1

        self.expected_output = expected_converted
        print(f"Changed Expected: {self.expected_output}")



if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    print(X)
    print("\n")
    print(y)
    print("-"*100)

    nn = NeuralNetwork(X, y)
    
    nn.add_layer(neurons=2)
    nn.add_layer(neurons=4)
    nn.add_layer(neurons=2)
    nn.display_layers()
    nn.backpropagation()
    