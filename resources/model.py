import os
import copy
import pickle

from flask_restful import Resource


class Model(Resource):

    def get(self):
        return {
            "directory": "./models/",
            "filename": "model.pickle"
        }


class ModelSerializer:

    @classmethod
    def serialize(cls, model):
        model_copy = copy.deepcopy(model)
        path = os.path.join(".", "models", "model.pickle")
        
        with open(path, "wb") as f:
            pickle.dump(model_copy, f)
