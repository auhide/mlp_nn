from sklearn.metrics import confusion_matrix


class ConfusionMatrix:
    
    @classmethod
    def confusion_mtx(cls, y_true, y_pred, labels=None):
        return confusion_matrix(y_true, y_pred, labels)


class Accuracy:

    @classmethod
    def accuracy(cls, y_true, y_pred):
        matches = [1 for y1, y2 in zip(y_true, y_pred) if y1 == y2]
        
        return round(len(matches) / len(y_true), 2)
        


class Evaluator(ConfusionMatrix, Accuracy):
    pass
