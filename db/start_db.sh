# Removing the previous image and container.
docker rm -f mongodb 
docker rmi mongodb:latest

docker build -t mongodb .
docker run -d --rm -p 27017-27019:27017-27019 --name mongodb mongodb