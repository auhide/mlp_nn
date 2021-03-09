FROM ubuntu:20.04

# Setuping Python and PIP
RUN apt-get update -y
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

# Going to the working directory (in the container)
WORKDIR /app

# Copy all local files to the container
COPY . /app

# Installing all needed modules
RUN pip3 install --no-cache-dir -r requirements.txt

# Exposing the 5000th container port
EXPOSE 5000

# Setting environment variables
ENV DATABASE "mongodb"
ENV DATASET_SIZE 100

# Migrating datasets & Starting the API 
ENTRYPOINT ["./setup_api.sh"]
