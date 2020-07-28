import numpy as np

from layers import Layer


class NeuralNetwork:
    # For testing purposes
    n_features = 2
    n_layers = 2
    neurons_per_hidden_layer = 5
    categories = 3

    def __init__(self, X, y):
        self.inputs = X
        self.expected_output = y

        self.layers = [
            Layer(self.n_features, self.neurons_per_hidden_layer),
        ]

        self.results = None


    def _initialize_hidden_layers(self):
        
        for _ in range(self.n_layers - 1):
            self.layers.append(Layer(self.neurons_per_hidden_layer, 
                                     self.neurons_per_hidden_layer))


    def forward_prop(self):
        self._initialize_hidden_layers()
        curr_inputs = self.inputs

        for i, layer in enumerate(self.layers):
            print(f"Layer[{i+1}] Neurons: {len(curr_inputs[0])}")
            print(curr_inputs.shape, "\n")

            curr_output = layer.forward(curr_inputs)
            curr_inputs = curr_output
        
        results_layer = Layer(self.neurons_per_hidden_layer,
                              self.categories)

        self.results = results_layer.forward(curr_inputs)
        print(f"Layer [{i+2}] Neurons: {len(self.results)}")
        self.cost()

        return self.results

    
    def convert_expected(self):
        """
        Converts the vector of expected results
        to a matrix, based on the neurons of the last layer.
        E.g. the neurons of the output layer are 3, the 
        expected results' vector will be converted from
        to m x 3, where m is the number of the expected results.
        """
        expected_converted = np.zeros((
            len(self.expected_output),
            len(self.results[0]) 
        ))


        for i, output in enumerate(self.expected_output):
            expected_converted[i][output] = 1

        self.expected_output = expected_converted


    def cost(self):
        self.convert_expected()
        
        errors_total = (1 / 2)* (self.expected_output - self.results) ** 2
        errors_total = errors_total.mean(axis=1)