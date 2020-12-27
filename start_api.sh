echo "Activating Virtual Environment"
source venv/bin/activate

echo "Starting the Flask API"
export FLASK_APP=api.py
export FLASK_ENV=development
flask run --host=0.0.0.0
