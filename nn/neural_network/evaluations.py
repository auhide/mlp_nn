from sklearn.metrics import confusion_matrix
import numpy as np


__all__ = ["Evaluator"]

class ConfusionMatrix:
    
    @classmethod
    def confusion_mtx(cls, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)


class Accuracy:

    @classmethod
    def accuracy(cls, y_true, y_pred):
        matches = [1 for y1, y2 in zip(y_true, y_pred) if y1 == y2]
        
        return round(len(matches) / len(y_true), 2)
        


class Evaluator(ConfusionMatrix, Accuracy):
    """
    Class inheriting functionalities of different evaluation classes
    """
    pass


class EpochEvaluator:

    def _evaluate_epoch(self, epoch):
        prediction = self._convert_as_prediction(
            self.predict(self.X)
        )
        self.epochs_accuracy[epoch] = Evaluator.accuracy(self.y, prediction)

    @staticmethod
    def _convert_as_prediction(raw_prediction):
        result = []

        for row in raw_prediction:
            result.append(list(row).index(max(row)))

        return np.array(result)