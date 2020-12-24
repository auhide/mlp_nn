import json

import numpy as np
from flask import Flask, request
from flask_restful import Resource, Api

from resources.resource import Architecture, Weights


app = Flask(__name__)
api = Api(app)


api.add_resource(Weights, "/weights")
api.add_resource(Architecture, "/architecture")


if __name__ == '__main__':
    app.run()