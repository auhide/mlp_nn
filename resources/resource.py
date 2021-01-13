import numpy as np
from flask import request
from flask_restful import Resource

from nn.overlays import NeuralNetFactory
from nn.neural_network.evaluations import Evaluator

from nn.neural_network.template_data import X, y
from preprocess.base import preprocess


class Architecture(Resource):

    def post(self):
        request_json = request.get_json(force=True)
        architecture, optimization, hyperparams = self._parse_request_json(
            request_json
        )

        X_train, X_test, y_train, y_test = preprocess(X, y)

        try:
            self.nn = NeuralNetFactory.define_nn(
                X=X_train, 
                y=y_train,
                architecture_dict=architecture,
                optimizer=optimization,
                **hyperparams
            )

            self.nn.fit()
            prediction = self._convert_as_prediction(
                self.nn.predict(X_test)
            )

            # Accuracy Evaluation
            accuracy = Evaluator.accuracy(y_test, prediction)
            epochs_accuracy = self.nn.epochs_accuracy

            # Confusion Matrix
            raw_confusion_matrix = Evaluator.confusion_mtx(y_test, prediction)
            confusion_matrix = self._convert_confusion_matrix(raw_confusion_matrix)

        except Exception as e:
            return {
                "StatusCode": 500,
                "Message": str(e)
            }

        weights = self._parse_nn_weights(self.nn._layers)

        return {
            "StatusCode": 200,
            "Message": "Successfully Created Model!",
            "Data": {
                "Weights": weights,
                "Accuracy": accuracy,
                "EpochsAccuracy": epochs_accuracy,
                "ConfusionMatrix": confusion_matrix,
            },
            "RequestData": request_json
        }

    @staticmethod
    def _parse_nn_weights(layers):
        parsed_weights_dict = {}

        for i, layer in enumerate(layers):
            weights = np.float32(layer.weights).tolist()
            parsed_weights_dict[i+1] = weights

        return parsed_weights_dict

    @staticmethod
    def _parse_request_json(json):
        """Casts the keys (number of neurons) to integers

        Args:
            json (dict): Architecture received from the POST request

        Returns:
            dict: Converted JSON
        """
        
        architecture = json["architecture"]
        optimization = json["optimization"]
        hyperparameters = json["hyperparameters"]

        return architecture, optimization, hyperparameters

    @staticmethod
    def _convert_as_prediction(raw_prediction):
        result = []

        for row in raw_prediction:
            result.append(list(row).index(max(row)))

        return np.array(result)

    @staticmethod
    def _convert_confusion_matrix(raw_matrix):
        converted_conf_mtx = []
        
        # Transposing the matrix and reversing each row.
        # This is done because that way it is easier to use the indexes of
        # the numpy array as x & y coordinates
        raw_matrix = np.flip(raw_matrix.T, 1)

        for i, row in enumerate(raw_matrix.T):
            
            for j, col in enumerate(row):
                single_element = {
                    "x": j, 
                    "y": i, 
                    "color": int(col)
                }
                converted_conf_mtx.append(single_element)

        return converted_conf_mtx