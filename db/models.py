import os
import numpy as np
from db.database import DatabaseClient


CSV_PATH = os.path.join("db", "csv_datasets")


# TODO: Add docstrings to these methods
class Dataset:
    
    def __init__(self, name, selected_features):
        self.name = name
        self.selected_features = selected_features

        self.features = self._generate_features()
        print(self.features)
        self.selected_feature_ids = self._get_features_ids()
        
        self.X, self.y = self._generate_dataset()

    def _get_features_ids(self):
        """Get the ids of the features that are in `feature_names`.

        Args:
            feature_names (list): A list of feature names that is used as a filter.

        Returns:
            list: The list of the feature ids.
        """
        selected_ids = []

        for name in self.selected_features:
            selected_ids.append(self.features[name])
        
        return selected_ids

    def get_feature_names(self):
        feature_names = []

        for feature_name, id_ in self.features_ids.items():
            
            # Adding every feature and skipping the label
            if id_ != max(self.features_ids.values()):
                feature_names.append(feature_name)

        return feature_names
    
    def _generate_features(self):
        _db_client = DatabaseClient()
        features = _db_client.get_dataset_metadata(
            name=self.name
        )["features_ids"]

        return features

    def _generate_dataset(self):
        csv_datasets = self._get_csvs()
        dataset_index = None

        print(csv_datasets)

        dataset_index = csv_datasets.index(f"{self.name}.csv")

        if dataset_index == None:
            raise Exception(f"Dataset with the name '{self.name}' does not exist!")

        X, y = self._read_dataset(self.name)
        
        return X, y
    
    def _read_dataset(self, dataset_name):
        dataset_path = os.path.join(CSV_PATH, f"{dataset_name}.csv")
        dataset_array = np.genfromtxt(dataset_path, delimiter=",")
        
        # Getting the Features (X) and the Labels (y)
        X, y = dataset_array[1:, :-1], dataset_array[1:, -1]
        
        # Filtering out only the selected features
        X = X[:, self.selected_feature_ids]

        return X, y

    @staticmethod
    def _get_csvs():
        return os.listdir(CSV_PATH)
