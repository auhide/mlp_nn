echo "Starting the tests..."

if [ -d 'venv' ]
then
    echo "Activating the Environment"
    source venv/bin/activate
else
    # Currently using virtualenv, might be better to use python3 -m ...
    echo "Setuping the Virtual Environment"
    virtualenv -p python3 venv
    pip install -r requirements.txt

    echo "Activating the Environment"
    source venv/bin/activate
fi

pip install coverage==5.5

coverage run --source=./nn -m unittest discover -s tests/ -p "*_tests.py"
coverage html