import numpy as np

from network.layers import Layer

LEARNING_RATE = 0.5
np.random.seed(0)


class NeuralNetwork:
    # For testing purposes
    n_features = 2
    n_layers = 1
    neurons_per_hidden_layer = 5
    epochs = 10000

    def __init__(self, X, y):
        self.inputs = X
        self.expected_output = y
        self.categories = len(set(y))
        self.results = None

        self.layers = [
            Layer(self.n_features, self.neurons_per_hidden_layer),
        ]
        self._initialize_hidden_layers()


    def _initialize_hidden_layers(self):
        
        for _ in range(self.n_layers):
            self.layers.append(Layer(self.neurons_per_hidden_layer, 
                                     self.categories))


    def forward_prop(self):
        self.layers[0].forward(self.inputs)
        self.results = self.layers[1].forward(self.layers[0].output)

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


    def predict(self):

        for curr_epoch in range(self.epochs):
            self.back_prop(curr_epoch)


    def back_prop(self, curr_epoch):
        
        if curr_epoch > 0:
            self.results = self.forward_prop()
            print("Result: ", self.results.shape)
            print(self.results)
        else:
            self.forward_prop()
        
        if curr_epoch == 0:
            self.convert_expected()

        print(f"Epoch: {curr_epoch}")

        print("\nBackpropagating...")
        for i in reversed(range(len(self.layers))):
            curr_layer = self.layers[i]

            print("Layer:", i)
            # Creating the deltas for the last layer
            if curr_layer == self.layers[-1]:
                curr_layer.errors = self.expected_output - self.results
                curr_layer.deltas = curr_layer.errors * curr_layer.activation_derivative()
                # print("Output Layer Deltas: ", curr_layer.deltas)

            else:
                next_layer = self.layers[i+1]

                curr_layer.errors = np.dot(next_layer.deltas, next_layer.weights.T)
                curr_layer.deltas = curr_layer.errors * curr_layer.activation_derivative()
                # print(curr_layer.deltas)

        self.update_weights()
        



    def update_weights(self):
        
        print("\nUpdating Weights...")
        for i in range(len(self.layers)):
            print("Layer:", i)
            curr_layer = self.layers[i]
            # print("Curr. Weights:", curr_layer.weights.shape)
            # print("Curr. Deltas:", curr_layer.deltas.shape)

            input_to_use = np.atleast_2d(self.inputs if i == 0 else self.layers[i-1].output)
            # print("Delta*Input Shape:", curr_layer.deltas.T.dot(input_to_use).shape)
            # print(f"Prev. Inputs: {input_to_use.shape}")
            change = curr_layer.deltas.T.dot(input_to_use) * LEARNING_RATE
            curr_layer.weights += change.T