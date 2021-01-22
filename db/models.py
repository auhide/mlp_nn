import pymongo as mongo
import numpy as np


class Dataset:
    
    def __init__(self, _id, name, data, fields_ids):
        self.name = name
        self.data = data
        self.fields_ids = fields_ids

    def to_numpy(self):
        full_dataset = np.array(self.data).T.astype(np.float128)
        X = full_dataset[:, :-1]
        y = full_dataset[:, -1]
        return {
            self.name: [X, y]
        }