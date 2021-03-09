
import os


DB_SERVER = os.environ.get("DATABASE", "mongodb")
DATASET_SIZE = os.environ.get("DATASET_SIZE", 100)