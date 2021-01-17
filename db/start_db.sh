docker rm mongodb
docker build -t mongodb .
docker run -p 27017-27019:27017-27019 --name mongodb mongodb
