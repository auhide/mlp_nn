from flask import request
from flask_restful import Resource

from db.database import DatabaseClient
from preprocess.base import shuffle_data

import numpy as np


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
        
        for doc in docs:
            names.append(doc["name"])
        
        for name in names:
            X, y = self._get_features(name)

        print(X, y)

        return {
            "result": names,
            "features": X.tolist(),
            "labels": y.astype(int).tolist()
        }

    def _get_features(self, name):
        X, y = self._db_client.get_dataset({ "name": "heart_disease" })
        X, y = shuffle_data(X, y)
        X = self._scale_features(X[:DSET_SIZE])

        return X, y[:DSET_SIZE]

    @staticmethod
    def _scale_features(X):
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float64)

        mds = MDS(2, random_state=0)
        X_2d = mds.fit_transform(X_scaled)

        return X_2d