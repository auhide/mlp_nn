import pymongo
from pymongo import collection

from config import ENV, DB_HOST, DB_PASSWORD, DB_CLUSTER


COLLECTION = "datasets"
DB_URI = f"mongodb+srv://admin:{DB_PASSWORD}@neuroad-cluster.avc2u.mongodb.net/{DB_CLUSTER}?retryWrites=true&w=majority"


class DatabaseClient:
    """A client that serves as a connection to a MongoDB.

    Args:
        server (string, optional): MongoDB server IP. Defaults to "mongodb".
        port (int, optional): MongoDB's exposed port. Defaults to 27017.
        db (string, optional): The database with which we communicate. Defaults to "nnvis-data".
    """

    def __init__(self, server=DB_HOST, port=27017, db=DB_CLUSTER):

        if ENV == "PROD":
            self.client = pymongo.MongoClient(DB_URI)
            
        else:
            self.client = pymongo.MongoClient(server, port)

        self.db = self.client[db]

    def get_collection(self, collection_name=COLLECTION):
        """Returns a collection based on its name.

        Args:
            collection_name (string, optional): The selected collection. Defaults to "datasets".

        Returns:
            pymongo.collection.Collection: MongoDB Collection
        """
        collection = self.db[collection_name]

        return collection

    def get_collection_documents(self, collection_name=COLLECTION):
        return self.db[collection_name].find()

    def get_dataset_metadata(self, name=None, collection_name=COLLECTION):
        return self.db[collection_name].find_one({
                "name": name
            }
        )

    def get_dataset_features(self, search_dict={}, collection_name=COLLECTION):
        doc = self.get_documents(collection_name, search_dict)[0]
        
        return doc


default_db_client = DatabaseClient()
