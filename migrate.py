"""
Inserts all initially needed datasets for the Neural Network API
"""

import os
import csv

from db.database import get_datasets_collection

CSV_PATH = os.path.join("db", "csv_datasets")


def migrate():
    """Insert the data from all csv files inside of `CSV_PATH` into MongoDB

    Returns:
        pymongo.collection.Collection: MongoDB Collection
    """

    csv_filenames = get_csvs()

    datasets = get_datasets_collection()

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
            headers = None

            for i, row in enumerate(reader):
                if i == 0:
                    headers = row

                else:
                    document_to_insert = {}
                    document_to_insert["name"] = name
                    
                    for n in range(len(headers)):
                        document_to_insert[headers[n]] = row[n]

                    mongo_collection.insert_one(document_to_insert)

        return True

    else:
        return False


if __name__ == "__main__":
    datasets = migrate()

    print(len(list(datasets.find())))