import pymongo

from db.models import Dataset

SERVER = "mongodb"
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

    def insert_documents(self, data, collection_name=COLLECTION):
        pass

    def get_documents(self, collection_name=COLLECTION, search_dict={}):
        """Get documents based on the collection - `collection_name` -  in which
        they are in and some keywords - `search_dict`

        Args:
            collection_name (string, optional): Name of the Collection. Defaults to "datasets".
            search_dict (dict, optional): Keywords used as a search query. Defaults to {}.

        Returns:
            numpy.array: The first found result in MongoDB, based on `search_dict`
        """
        result = self.db[collection_name].find(search_dict)

        return list(result)

    def get_dataset(self, search_dict={}, collection_name=COLLECTION):
        """Returns the features (X) and labels (y) of the first matched dataset 
        based on `search_dict`.

        Args:
            search_dict (dict, optional): Keywords used as a search query. Defaults to {}.
            collection_name (string, optional): Name of the Collection. Defaults to "datasets".

        Returns:
            tuple: A tuple representing the features and labels - X and y
        """
        doc = self.get_documents(collection_name, search_dict)[0]
        dataset = Dataset(**doc).to_numpy()
        X, y = dataset[search_dict["name"]][0], dataset[search_dict["name"]][1]

        return X, y

    def get_collection(self, collection_name=COLLECTION):
        """Returns a collection based on its name.

        Args:
            collection_name (string, optional): [description]. Defaults to "datasets".

        Returns:
            pymongo.collection.Collection: MongoDB Collection
        """
        collection = self.db[collection_name]

        return collection


default_db_client = DatabaseClient()


if __name__ == "__main__":

    # Example code, you should run that locally, hence - the server string:
    db_client = DatabaseClient(server="127.0.0.1")
    X, y = db_client.get_dataset({ "name": "iris" })

