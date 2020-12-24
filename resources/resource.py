import numpy as np
from flask import request
from flask_restful import Resource

from nn.overlays import NeuralNetFactory
from nn.neural_network.evaluations import Evaluator

from nn.neural_network.template_data import X, y
from preprocess.base import preprocess

GLOBAL_NN = None


class Weights(Resource):

    @staticmethod
    def _parse_nn_layers(layers):
        parsed_weights_dict = {}

        for i, layer in enumerate(layers):
            weights = np.float32(layer.weights).tolist()
            parsed_weights_dict[i+1] = weights

        return parsed_weights_dict

    def get(self):
        print(GLOBAL_NN)
        
        if GLOBAL_NN:
            return self._parse_nn_layers(GLOBAL_NN._layers)
        
        return {
            "StatusCode": 404,
            "Message": "You haven't created a Neural Network"
        }


class Architecture(Resource):

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

    def post(self):
        global GLOBAL_NN
        request_json = request.get_json(force=True)
        architecture, optimization, hyperparams = self._parse_request_json(
            request_json
        )

        X_train, X_test, y_train, y_test = preprocess(X, y)

        try:
            GLOBAL_NN = NeuralNetFactory.define_nn(
                X=X_train, 
                y=y_train,
                architecture_dict=architecture,
                optimizer=optimization,
                **hyperparams
            )

            GLOBAL_NN.fit()
            print("Training Has Finished\n")
            prediction = GLOBAL_NN.predict(X_test)
            print("Prediction:\n", prediction)
            print("Expected:\n", y_test)

            accuracy = Evaluator.accuracy(y_test, prediction)
            print("\nAccuracy: ", accuracy)

        except Exception as e:
            return {
                "StatusCode": 500,
                "Message": str(e)
            }

        return {
            "StatusCode": 200,
            "Message": "Successfully Created NN"
        }