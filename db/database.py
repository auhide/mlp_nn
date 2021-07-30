import pymongo
from pymongo import collection

from config import DB_SERVER


SERVER = DB_SERVER
DATABASE = "nnvis-data"
COLLECTION = "datasets"


class DatabaseClient:
    """A client that serves as a connection to a MongoDB.

    Args:
        server (string, optional): MongoDB server IP. Defaults to "mongodb".
        port (int, optional): MongoDB's exposed port. Defaults to 27017.
        db (string, optional): The database with which we communicate. Defaults to "nnvis-data".
    """

    def __init__(self, server=SERVER, port=27017, db=DATABASE):
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
