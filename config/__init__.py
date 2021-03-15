
import os


# The Docker container is passing 'mongodb' as a DATABASE env. variable.
DB_SERVER = os.environ.get("DATABASE", "127.0.0.1")
DATASET_SIZE = int(os.environ.get("DATASET_SIZE", 150))
DISPLAYED_DATASET_SIZE = int(os.environ.get("DATASET_SIZE", 100))