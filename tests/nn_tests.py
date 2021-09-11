import unittest

import numpy as np

from nn.overlays import NeuralNetFactory
from nn.neural_network.exceptions import WrongLayerFormat


class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        # The XOR function
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = [0, 1, 1, 0]

        self.architecture = {
            1: 3, 
            2: 2,
        }
        
    def test_architecture_generation(self):
        nn = NeuralNetFactory.define_nn(
            architecture_dict=self.architecture, 
            X=self.X, 
            y=self.y
        )
        expected_arch = {0: 2, 1: 3, 2: 2}
        self.assertTrue(nn.get_architecture() == expected_arch)

    def test_wrong_architecture(self):
        # The architecture defined here has 3 output neurons, but the data
        # has 2 labels
        nn = NeuralNetFactory.define_nn(
            architecture_dict={1: 3, 2: 2, 3: 3}, 
            X=self.X, 
            y=self.y
        )

        with self.assertRaises(WrongLayerFormat):
            nn.fit()

    def train(self, X_test, optimizer):
        nn = NeuralNetFactory.define_nn(
            optimizer=optimizer,
            batch=2,
            architecture_dict=self.architecture, 
            X=self.X, 
            y=self.y,
            epochs=300,
            learning_rate=0.1
        )
        
        nn.fit()
        prediction = nn.predict(X_test)[0]

        # Getting the number representing the prediction
        prediction = list(prediction).index(max(prediction))

        return prediction

    def test_training_sgd(self):
        expected_prediction = 1
        X_test = np.array([[0, 1]])

        prediction = self.train(X_test=X_test, optimizer="sgd")

        self.assertTrue(prediction == expected_prediction)

    def test_training_sgdm(self):
        expected_prediction = 0
        X_test = np.array([[0, 0]])

        prediction = self.train(X_test=X_test, optimizer="sgdm")
        
        self.assertTrue(prediction == expected_prediction)

    def test_training_adagrad(self):
        expected_prediction = 1
        X_test = np.array([[1, 0]])

        prediction = self.train(X_test=X_test, optimizer="adagrad")
        
        self.assertTrue(prediction == expected_prediction)

    def test_training_adam(self):
        expected_prediction = 0
        X_test = np.array([[1, 1]])

        prediction = self.train(X_test=X_test, optimizer="adam")
        
        self.assertTrue(prediction == expected_prediction)
