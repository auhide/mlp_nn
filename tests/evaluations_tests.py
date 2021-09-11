import unittest

import numpy as np

from nn.neural_network.evaluations import Evaluator


class TestEvaluator(unittest.TestCase):

    def test_accuracy(self):
        y_true = [0, 0, 1]
        y_pred = [0, 1, 1]
        accuracy = Evaluator.accuracy(y_pred=y_pred, y_true=y_true)
        
        self.assertTrue(accuracy == 0.67)

    def test_confusion_matrix(self):
        expected_mtx = np.array(
            [
                [1, 1], 
                [0, 1]
            ]
        )

        conf_mtx = Evaluator.confusion_mtx(
            y_true=[0, 0, 1], 
            y_pred=[0, 1, 1]
        )
        self.assertTrue((expected_mtx==conf_mtx).all())
