import numpy as np


class Dataset:
    
    def __init__(self, _id, name, data, features_ids):
        self.name = name
        self.data = data
        self.features_ids = features_ids

    def to_numpy(self):
        full_dataset = np.array(self.data).T.astype(np.float128)
        X = full_dataset[:, :-1]
        y = full_dataset[:, -1]
        
        return {
            self.name: [X, y]
        }

    def to_numpy_by_features(self, feature_names):
        full_dataset = np.array(self.data).T.astype(np.float128)
        features_ids = self.get_features_ids(feature_names)

        if len(features_ids) == 1 and features_ids[0] == "all":
            X = full_dataset[:, :-1]
        
        else:
            X = full_dataset[:, features_ids]

        y = full_dataset[:, -1]
        
        return {
            self.name: [X, y]
        }

    def get_features_ids(self, feature_names):
        """Get the ids of the features that are in `feature_names`.

        Args:
            feature_names (list): A list of feature names that is used as a filter.

        Returns:
            list: The list of the feature ids.
        """
        selected_ids = []

        for name in feature_names:
            selected_ids.append(self.features_ids[name])
        
        return selected_ids