from flask import request
from flask_restful import Resource

from db.database import DatabaseClient

DEV = True

DB_DEV_SERVER = "127.0.0.1"
DB_PROD_SERVER = "mongodb"

if DEV:
    DB_SERVER = DB_DEV_SERVER

else:
    DB_SERVER = DB_PROD_SERVER


class Datasets(Resource):

    def get(self):
        db_client = DatabaseClient(server=DB_SERVER)
        docs = db_client.get_documents()
        
        names = []
        
        for doc in docs:
            names.append(doc["name"])

        return {
            "result": names
        }