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
        """Create a numpy arrays of X and y and filter out the X columns based
        on `feature_names`.

        Args:
            feature_names (list): The names of the selected features.

        Returns:
            dict: A dictionary with a key - the name of the dataset and value
            the features (X) and label (y).
        """
        full_dataset = np.array(self.data).T.astype(np.float128)

        # Selecting all features
        if len(feature_names) == 1 and feature_names[0] == "all":
            X = full_dataset[:, :-1]
        
        # Selecting certain features that are in the list of feature_names
        else:
            features_ids = self.get_features_ids(feature_names)
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

    def get_feature_names(self):
        feature_names = []

        for feature_name, id_ in self.features_ids.items():
            
            # Adding every feature and skipping the label
            if id_ != max(self.features_ids.values()):
                feature_names.append(feature_name)

        return feature_names