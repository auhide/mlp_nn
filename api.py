from resources.pca import PrincipalComponentAnalysis
from flask import Flask, send_file, safe_join
from flask_cors import CORS
from flask_restful import  Api

from config import DEBUG, FLASK_HOST, FLASK_PORT
from resources.architecture import Architecture
from resources.datasets import Datasets, DatasetsNames, DatasetsInformation
from resources.pca import PrincipalComponentAnalysis
from resources.model import Model


app = Flask(__name__)
cors = CORS(app)
api = Api(app)

api.add_resource(Architecture, "/architecture")

api.add_resource(Datasets, "/datasets/<string:dataset_name>")
api.add_resource(DatasetsNames, "/datasets/names")
api.add_resource(DatasetsInformation, "/datasets/info/<string:dataset_name>")

api.add_resource(PrincipalComponentAnalysis, "/pca")

api.add_resource(Model, "/model")


# Using this method for the download of the model.
@api.representation('application/octet-stream')
def output_file(data, code, headers):
    filepath = safe_join(data["directory"], data["filename"])

    response = send_file(
        filename_or_fp=filepath,
        mimetype="application/octet-stream",
        as_attachment=True,
        attachment_filename=data["filename"]
    )
    return response


if __name__ == '__main__':
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=DEBUG)
