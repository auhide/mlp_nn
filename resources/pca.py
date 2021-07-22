import numpy as np
from flask import request
from flask_restful import Resource

from config import DB_SERVER
from preprocess.base import normalize_data
from db.database import DatabaseClient


class PcaTransformer:

    def transform(self, X, n_components, feature_names):
        X = normalize_data(X)
        print("Normalized:")
        print(X)

        # Center data (subtract the mean off it)
        X_centered = self._center_data(X)
        X_cov = self._calc_covariance_matrix(X_centered).astype("float64")
        
        # Calculate the Eigenvectors & Eigenvalues
        eig_values, eig_vectors = np.linalg.eig(X_cov)
        print(eig_vectors)
        print("\n\n")
        print(eig_values)
        
        top_n_features, principal_vectors = self._get_top_n_components(
            evectors=eig_vectors,
            evalues=eig_values,
            n=n_components,
            features=feature_names
        )

        principal_components_matrix = X_centered.dot(principal_vectors.T)

        return top_n_features, principal_components_matrix

    @staticmethod
    def _get_top_n_components(evectors, evalues, n, features):
        vector_magnitudes = []
        principal_vectors = []

        # Generating a tuple of the indices and the magnitude of each vector
        for i, vector in enumerate(evectors.T):
            current_feature_vector = evalues[i] * vector
            magnitude = np.linalg.norm(current_feature_vector)
            vector_magnitudes.append(
                (i, magnitude)
            )

        # Sorting the list of tuples by the second element
        vector_magnitudes.sort(key=lambda x: x[1], reverse=True)
        print("By Magnitude:")
        print(vector_magnitudes)

        top_n_features = []
        # Filtering out the first `n` eigenvectors sorted by magnitude
        for i in range(n):
            feature_index, vec_magnitude = vector_magnitudes[i]
            top_n_features.append(features[feature_index])

            # Adding the vectors to a matrix
            principal_vectors.append(evectors[feature_index])

        principal_vectors = np.array(principal_vectors)

        return top_n_features, principal_vectors

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


class PrincipalComponentAnalysis(Resource):

    def post(self):
        # Parsing the request
        request_json = request.get_json(force=True)
        dataset_name, features, n_components = self._parse_request_json(request_json)

        # Getting the dataset
        db_client = DatabaseClient(server=DB_SERVER)
        X, _ = db_client.get_dataset({ "name": dataset_name }, features=features)
        pca_transformer = PcaTransformer()
        
        # Transforming the dataset into an array of principle components
        top_n_features, principal_components_matrix = pca_transformer.transform(
            X=X, 
            n_components=n_components,
            feature_names=features
        )
        print(principal_components_matrix)
        print(top_n_features)

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
