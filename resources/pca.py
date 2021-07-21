import numpy as np
from flask import request
from flask_restful import Resource

from config import DB_SERVER
from db.database import DatabaseClient


class PrincipalComponentAnalysis(Resource):

    def post(self):
        # Parsing the request
        request_json = request.get_json(force=True)
        dataset_name, features, n_components = self._parse_request_json(request_json)

        # Getting the dataset
        db_client = DatabaseClient(server=DB_SERVER)
        X, _ = db_client.get_dataset({ "name": dataset_name }, features=features)

        # Center data (subtract the mean off it)
        X_centered = self._center_data(X)
        X_cov = self._calc_covariance_matrix(X_centered).astype("float64")
        
        # Calculate the Eigen vectors & values
        eig_values, eig_vectors = np.linalg.eig(X_cov)
        print(eig_vectors)
        print("\n\n")
        print(eig_values)

        return request_json

    @staticmethod
    def _parse_request_json(request):
        """Gets the needed data that's coming from the request.

        Args:
            request (dict): The raw request

        Returns:
            tuple: A tuple of the needed data (dataset_name, features, n_components).
        """
        dataset_name = request["dataset_name"]
        features = request["features"]
        n_components = request["n_components"]

        return dataset_name, features, n_components

    @staticmethod
    def _center_data(X):
        """Centers the data around the mean.
        This method basically subtracts the mean off the matrix `X`.

        Args:
            X (numpy.matrix): The dataset matrix.

        Returns:
            numpy.matrix: The centered version of `X`.
        """
        # Get the mean row
        mean_row = X.mean(0)
        ones_matrix = np.ones(X.shape)

        # Matrix with the shape of X, constructed of only mean rows
        mean_matrix = ones_matrix * mean_row
        X_centered = X - mean_matrix

        return X_centered

    @staticmethod
    def _calc_covariance_matrix(X):
        """Calculates the covariance matrix of `X`.

        Args:
            X (numpy.matrix): A matrix, with no restrictions in regards of its dimensions.

        Returns:
            numpy.matrix: An NxN symmetric covariance matrix.
        """
        return X.T.dot(X)
