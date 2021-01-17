import pymongo


def get_datasets_collection():
    """Returns a collection that consists of all datasets.

    Returns:
        pymongo.collection.Collection: MongoDB Collection
    """
    # Connecting to the Database and selecting the Collection
    
    client = pymongo.MongoClient("mongodb", 27017)
    db = client["nnvis-data"]
    datasets = db["datasets"]

    return datasets