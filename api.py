import json

import numpy as np
from flask import Flask, request
from flask_cors import CORS
from flask_restful import Resource, Api

from resources.resource import Architecture


app = Flask(__name__)
cors = CORS(app)
api = Api(app)

api.add_resource(Architecture, "/architecture")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
