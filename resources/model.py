import os
import copy
import pickle

import numpy
from flask_restful import Resource
from nn.neural_network.activations import NeuronActivations, ACTIVATIONS


class Model(Resource):

    def get(self):
        return {
            "directory": "./db/csv_datasets/",
            "filename": "gender_voice.csv"
        }


class ModelSerializer:

    @classmethod
    def _save_model(cls, model):
        model_copy = copy.deepcopy(model)
        path = os.path.join(".", "models", "model.pickle")
        
        print(locals())
        with open(path, "wb") as f:
            pickle.dump(model_copy, f)


class NeuroadNetwork(NeuronActivations):

    def __init__(self, weights, activation):
        self.weights = weights
        self.activation = activation

    def predict(self, X):
        curr_x = numpy.array(X)
        
        for curr_weights in self.weights:
            weights = numpy.array(curr_weights)
            curr_x = curr_x.dot(weights)

            # Using the activation function
            curr_x = eval(f"self.{ACTIVATIONS[self.activation]}_activation(curr_x)")

        prediction = self._convert_as_prediction(curr_x)
        
        return prediction

    @staticmethod
    def _convert_as_prediction(raw_prediction):
        result = []

        for row in raw_prediction:
            result.append(list(row).index(max(row)))

        return numpy.array(result)
