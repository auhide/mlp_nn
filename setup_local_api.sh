echo "Starting the Mongo Database..."
cd db/
./start_db.sh &

echo "Inserting the default datasets..."
cd ../

if [ -d 'venv' ]
then
    echo "Activating the Environment"
    source venv/bin/activate
else
    # The machine has to have virtualenv set up.
    # I had local problems with python -m venv, so I decided to use virtualenv.
    echo "Setuping the Virtual Environment"
    virtualenv -p python3 venv

    echo "Activating the Environment"
    source venv/bin/activate
fi

python migrate.py

echo "Starting the API..."
python api.py