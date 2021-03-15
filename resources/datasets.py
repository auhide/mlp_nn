import numpy as np
from flask import request
from flask_restful import Resource
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import MDS

from config import DB_SERVER, DISPLAYED_DATASET_SIZE
from db.database import DatabaseClient
from preprocess.base import shuffle_data


class Datasets(Resource):
    
    def get(self, dataset_name):
        self._db_client = DatabaseClient(server=DB_SERVER)
        docs = self._db_client.get_documents()
        
        chosen_dataset = self._get_features(dataset_name)

        return chosen_dataset

    def _get_features(self, name):
        X, y = self._db_client.get_dataset({ "name": name })
        X, y = shuffle_data(X, y)
        X = self._scale_features(X[:DISPLAYED_DATASET_SIZE])
        features_to_visualize = self._format_features(X, y[:DISPLAYED_DATASET_SIZE])

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


class DatasetsNames(Resource):

    def get(self):
        self._db_client = DatabaseClient(server=DB_SERVER)
        docs = self._db_client.get_documents()
        
        names = []
        
        for doc in docs:
            presentable_name = self._prepare_presentable_name(doc["name"])
            names.append({
                "presentable_name": presentable_name,
                "name": doc["name"]
            })

        return names

    @staticmethod
    def _prepare_presentable_name(name):
        presentable_name = [
            sub_name.capitalize() 
            for sub_name in name.split("_")
        ]
        presentable_name = " ".join(presentable_name)

        return presentable_name
