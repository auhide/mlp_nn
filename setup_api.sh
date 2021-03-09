#!/bin/bash

echo "Starting the Mongo Database"
cd db/
./start_db.sh
cd ..

echo "Migrating Datasets..."
python3 migrate.py

echo "Starting the API..."
python3 api.py
