import numpy as np
from flask import request
from flask_restful import Resource
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import MDS

from db.database import DatabaseClient
from preprocess.base import shuffle_data



DEV = True

DB_DEV_SERVER = "127.0.0.1"
DB_PROD_SERVER = "mongodb"

if DEV:
    DB_SERVER = DB_DEV_SERVER

else:
    DB_SERVER = DB_PROD_SERVER

# This will be sent to the UI for Dataset visualization
DSET_SIZE = 100


class Datasets(Resource):
    def get(self):
        self._db_client = DatabaseClient(server=DB_SERVER)
        docs = self._db_client.get_documents()
        
        names = []
        datasets_result = {}
        
        for doc in docs:
            names.append(doc["name"])
        
        for name in names:
            data_to_visualize = self._get_features(name)
            datasets_result[name] = data_to_visualize

        return datasets_result

    def _get_features(self, name):
        X, y = self._db_client.get_dataset({ "name": name })
        X, y = shuffle_data(X, y)
        X = self._scale_features(X[:DSET_SIZE])
        features_to_visualize = self._format_features(X, y[:DSET_SIZE])

        return features_to_visualize

    @staticmethod
    def _scale_features(X):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float64)

        mds = MDS(2, random_state=0)
        X_2d = mds.fit_transform(X_scaled)

        return X_2d

    @staticmethod
    def _format_features(X, y):
        formatted_result = []

        for i, features in enumerate(X):
            formatted_result.append({
                "x": features[0],
                "y": features[1],
                "class": int(y[i])
            })

        return formatted_result
