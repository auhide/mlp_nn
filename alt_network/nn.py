from template_data import X, y
from layers import Layer
from exceptions import WrongLayerFormat



class NeuralNetwork:

    def __init__(self, X, y, learning_rate=0.5):
        self.X = X
        self.y = y

        self.input_neurons = len(X[0])
        self.output_neurons = len(set(y))
        self.l_rate = learning_rate

        self.layers = []


    def forward(self):
        print(f"Input Neurons: {self.input_neurons}")
        print(f"Output Neurons {self.output_neurons}")

        if not self._output_layer_is_valid():
            raise WrongLayerFormat(f"Last layer's neurons have to be {self.output_neurons}")

        for i, layer in enumerate(self.layers):
            
            if i == 0:
                prev_output = layer.forward(X)
            
            else:
                prev_output = layer.forward(prev_output)


    def backpropagation(self):
        self.forward()
        self.gradient_descent()


    def get_deltas(self):
        pass


    def gradient_descent(self):

        for layer in reversed(self.layers):
            
            if layer == self.layers[-1]:
                


    def _output_layer_is_valid(self):
        """True if the expected output layer's neurons are equal to the latest \
        added layer's neurons, else - False

        Returns:
            bool: Whether the out layer is valid or not
        """
        return self.layers[len(self.layers)-1].n_neurons == self.output_neurons

    
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

    
    def display_layers(self):
        print(f"Layer[0]: (0, {self.input_neurons})")

        for i in range(len(self.layers)):
            print(f"Layer[{i+1}]: ({self.layers[i].n_inputs}, {self.layers[i].n_neurons})")
            


if __name__ == "__main__":
    print(X)
    print("\n")
    print(y)
    print("-"*100)

    nn = NeuralNetwork(X, y)
    
    nn.add_layer(neurons=3)
    nn.add_layer(neurons=3)
    nn.display_layers()
    nn.backpropagation()
    