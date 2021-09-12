"""
Inserts all initially needed info about the datasets for the Neural Network API
"""
import os
import csv

from db.database import default_db_client


CSV_PATH = os.path.join("db", "csv_datasets")


def migrate():
    """Insert the data from all csv files inside of `CSV_PATH` into MongoDB

    Returns:
        pymongo.collection.Collection: MongoDB Collection
    """

    csv_filenames = get_csvs()

    datasets = default_db_client.get_collection("datasets")

    # Writing each csv file to the MongoDB
    for filename in csv_filenames:
        dataset_name = filename.split(".")[0]

        csv_to_mongo(
            dataset_name,
            os.path.join(CSV_PATH, filename),
            datasets
        )

    for row in datasets.find():
        print(row)

    return datasets


def get_csvs():
    return os.listdir(CSV_PATH)


def csv_to_mongo(name, filename, mongo_collection):
    """Converts a CSV file rows to MongoDB documents.

    Args:
        name (string): The name field of the documents
        filename (string): Filename of the CSV file
        mongo_collection (pymongo.collection.Collection): A MongoDB Collection

    Returns:
        bool: Boolean, whether the conversion is successful or not
    """

    if filename:

        with open(filename, "r") as f:
            reader = csv.reader(f)
            # Reading the Headers of the dataset
            headers = next(reader)
            document_to_insert = {}
            document_to_insert["name"] = name
            document_to_insert["features_ids"] = {}

            for i, header in enumerate(headers):
                if i != len(headers) - 1:
                    # Add headers' field to the DB
                    document_to_insert["features_ids"][header] = i

            mongo_collection.insert_one(document_to_insert)

        return True

    else:
        return False


if __name__ == "__main__":
    datasets = migrate()
