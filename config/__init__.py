
import os


ENV = os.environ.get("ENV", "DEV")

FLASK_HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.environ.get("PORT", 5000))

DEBUG = os.environ.get("DEBUG", True)

# The Docker container is passing 'mongodb' as a DATABASE env. variable.
DB_SERVER = os.environ.get("DATABASE", "127.0.0.1")
DATASET_SIZE = int(os.environ.get("DATASET_SIZE", 150))
DISPLAYED_DATASET_SIZE = int(os.environ.get("DATASET_SIZE", 100))
