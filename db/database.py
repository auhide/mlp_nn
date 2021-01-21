import pymongo


SERVER = "mongodb"
DATABASE = "nnvis-data"


class DatabaseClient:

    def __init__(self, server=SERVER, port=27017, db=DATABASE):
        self.client = pymongo.MongoClient(server, port)
        self.db = self.client[db]

    def insert_documents(self, collection_name, data):
        pass

    def get_documents(self, collection_name):
        result = self.db[collection_name].find()

        return list(result)

    def get_collection(self, collection_name):
        collection = self.db[collection_name]

        return collection


default_db_client = DatabaseClient()