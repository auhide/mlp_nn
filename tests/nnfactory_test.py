"""
"""

import unittest
from nn.overlays import (
    neuralnet_with_optimizer,
    NeuralNetFactory,
    NeuralNetwork,
    AdamOptimizer
)
from nn.neural_network.exceptions import OptimizerDoesNotExist


class TestNeuralNetFactory(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Add attributes that will be used in multiple tests
    
    def test_valid_neuralnet_descendant(self):
        """
        Whether the generated NeuralNet object from the NeuralNetFactory
        is a descendant of the class - NeuralNetwork.
        """
        self.assertTrue(
            isinstance(
                NeuralNetFactory.get_nn()(X=[[0]], y=[0]), 
                NeuralNetwork
            )
        )

    def test_valid_optimizer(self):
        """
        Whether the optimizer is correctly set or not.
        """
        nn = NeuralNetFactory.get_nn("adam")
        self.assertTrue(AdamOptimizer in nn.__bases__)

    def test_invalid_optimizer(self):
        """
        Whether a certain custom exception is raised, when a non-existent 
        optimizer is set, or not.
        """
        with self.assertRaises(OptimizerDoesNotExist):
            NeuralNetFactory.get_nn("imaginary_optimizer")