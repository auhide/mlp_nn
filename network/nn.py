import numpy as np

from network.layers import Layer



class NeuralNetwork:

    def __init__(self, X, y, **kwargs):
        self.__dict__.update(kwargs)
        
        self.inputs = X
        self.expected_output = y
        self.categories = len(set(y))
        self.results = None

        # Adding the first layer
        self.layers = [
            Layer(len(X[0]), self.hidden_neurons, random=self.random),
        ]
        if self.hidden_layers > 1:
            self._initialize_hidden_layers()


    def _initialize_hidden_layers(self):
        
        for i in range(self.hidden_layers - 2):
            self.layers.append(Layer(self.hidden_neurons, 
                                     self.hidden_neurons, 
                                     self.random))
            
        self.layers.append(Layer(self.hidden_neurons, 
                                 self.categories, 
                                 self.random))

    
    def forward_prop(self, layer_input):

        for i in range(self.hidden_layers):
            # print("Forwarding Layer: ", i)
            self.layers[i].forward(layer_input)
            layer_input = self.layers[i].output

        self.results = layer_input

        return self.results

    
    def _convert_expected(self):
        """
        Converts the vector of expected results
        to a matrix, based on the neurons of the last layer.

        E.g. the neurons of the output layer are 3, the 
        expected results' vector will be converted from
        to mx3, where m is the number of the expected results.
        """
        expected_converted = np.zeros((
            len(self.expected_output),
            len(self.results[0]) 
        ))


        for i, output in enumerate(self.expected_output):
            expected_converted[i][output] = 1

        self.expected_output = expected_converted


    def predict(self):
        curr_epoch = 0

        while True:
            self._back_prop(curr_epoch)
            curr_rmse = self.get_rmse()
            curr_mse = self.get_mse()
            
            print(f"epoch: {curr_epoch}; rmse: {curr_rmse}, mse: {curr_mse}")
            if curr_rmse < self.rmse_threshold:
                break

            curr_epoch += 1

        return self, self.layers[-1].output


    def _back_prop(self, curr_epoch):
        
        if curr_epoch > 0:
            self.results = self.forward_prop(self.inputs)
            # print("Result: ", self.results.shape)
            # print(self.results)
        else:
            self.forward_prop(self.inputs)
        
        if curr_epoch == 0:
            self._convert_expected()


        # print("\nBackpropagating...")
        for i in reversed(range(len(self.layers))):
            curr_layer = self.layers[i]

            # print("Layer:", i)
            # Creating the deltas for the last layer
            if curr_layer == self.layers[-1]:
                curr_layer.errors = self.expected_output - self.results
                curr_layer.deltas = curr_layer.errors * curr_layer.activation_derivative()

            else:
                next_layer = self.layers[i+1]

                curr_layer.errors = np.dot(next_layer.deltas, next_layer.weights.T)
                curr_layer.deltas = curr_layer.errors * curr_layer.activation_derivative()

        self._update_weights()
        

    def _update_weights(self):
        
        # print("\nUpdating Weights...")
        for i in range(len(self.layers)):
            # print("Layer:", i)
            curr_layer = self.layers[i]

            input_to_use = np.atleast_2d(self.inputs if i == 0 else self.layers[i-1].output)

            change = curr_layer.deltas.T.dot(input_to_use) * self.l_rate
            curr_layer.weights += change.T


    def get_rmse(self):
        output_matrix = self.layers[-1].output

        return np.sqrt(np.mean((output_matrix - self.expected_output) ** 2))


    def get_mse(self):
        output_matrix = self.layers[-1].output

        return np.mean((output_matrix - self.expected_output) ** 2)