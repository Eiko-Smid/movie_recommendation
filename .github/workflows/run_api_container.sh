#!/bin/bash

# Define container name
CONTAINER_NAME="movie-reco-api-ci"

echo "Removing old container if exists"
docker rm -f $CONTAINER_NAME 2>/dev/null

echo "Building API image"
docker build -f Dockerfile.api -t movie-reco-api:ci .

echo "Run container"
docker run -d \
  --name $CONTAINER_NAME \
  -e TESTING=true \
  -p 8000:8000 \
  movie-reco-api:ci

echo "Waiting for startup..."
sleep 5

echo "Checking if container is running"
if [ "$(docker inspect -f '{{.State.Running}}' $CONTAINER_NAME)" != "true" ]; then
  echo "Container crashed!"
  docker logs $CONTAINER_NAME
  exit 1
fi

echo "Logs:"
docker logs $CONTAINER_NAME

echo "Testing health endpoint until reachable"
for i in {1..10}; do
  if curl -s http://localhost:8000/health; then
    echo ""
    echo "Health endpoint reachable!"
    break
  fi
  echo "Waiting..."
  sleep 2
done

echo "Stop container"
docker stop $CONTAINER_NAME

echo "Done!"