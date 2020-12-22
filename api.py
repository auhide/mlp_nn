from flask import Flask, jsonify, request
from flask_restful import Resource, Api

from nn.overlays import NeuralNetFactory
from nn.neural_network.evaluations import Evaluator

# Currently used for testing purposes
from nn.neural_network.template_data import X, y
from preprocess.base import preprocess


app = Flask(__name__)
api = Api(app)


class Weights(Resource):

    def get(self):
        pass


class Architecture(Resource):

    @staticmethod
    def _parse_architecture_json(json):
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
        architecture = request.get_json(force=True)
        architecture, optimization, hyperparams = self._parse_architecture_json(
            architecture
        )

        X_train, X_test, y_train, y_test = preprocess(X, y)

        try:
            nn = NeuralNetFactory.define_nn(
                X=X_train, 
                y=y_train,
                architecture_dict=architecture,
                optimizer=optimization,
                **hyperparams
            )

            nn.fit()
            print("Training Has Finished\n")
            prediction = nn.predict(X_test)
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


api.add_resource(Weights, "/weights")
api.add_resource(Architecture, "/architecture")

if __name__ == '__main__':
    app.run(debug=True)