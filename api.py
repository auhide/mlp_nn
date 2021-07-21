from resources.pca import PrincipalComponentAnalysis
from flask import Flask
from flask_cors import CORS
from flask_restful import  Api

from resources.architecture import Architecture
from resources.datasets import Datasets, DatasetsNames, DatasetsInformation
from resources.pca import PrincipalComponentAnalysis


app = Flask(__name__)
cors = CORS(app)
api = Api(app)

api.add_resource(Architecture, "/architecture")

api.add_resource(Datasets, "/datasets/<string:dataset_name>")
api.add_resource(DatasetsNames, "/datasets/names")
api.add_resource(DatasetsInformation, "/datasets/info/<string:dataset_name>")
api.add_resource(PrincipalComponentAnalysis, "/pca")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
