import unittest

import numpy as np

from nn.neural_network.layers import Layer


class TestLayer(unittest.TestCase):

    def setUp(self):
        self.layer = Layer(
            n_inputs=3, 
            n_neurons=5,
            random_state=0,
            activation="sigm"
        )

    def test_layer_matrices_dimensions(self):
        self.assertTrue(
            self.layer.weights.shape == (3, 5) and \
            self.layer.biases.shape == (1, 5)
        )

    def test_layer_forward_prop(self):
        x = np.array(
            [0, 1, 0]
        )
        self.layer.forward(input_=x)

    def test_layer_wrong_input_dims(self):
        x = np.array(
            [0, 1, 0, 1]
        )
        
        with self.assertRaises(ValueError):
            self.layer.forward(input_=x)
