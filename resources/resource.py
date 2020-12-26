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

            accuracy = Evaluator.accuracy(y_test, prediction)

        except Exception as e:
            return {
                "StatusCode": 500,
                "Message": str(e)
            }

        weights = self._parse_nn_weights(self.nn._layers)

        return {
            "StatusCode": 200,
            "Message": "Successfully Created Neural Network",
            "Data": {
                "Weights": weights,
                "Accuracy": accuracy,
            }
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
        # print(raw_prediction)

        for row in raw_prediction:
            result.append(list(row).index(max(row)))

        return np.array(result)
