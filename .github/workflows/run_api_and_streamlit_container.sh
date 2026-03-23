#!/bin/bash

# Define API params
CONTAINER_NAME_API="movie-reco-api-ci"
IMAGE_NAME_API="movie-reco-api:ci"
HEALTH_ENDPOINT_API="http://localhost:8000/health"
INTERNAL_HEALTH_ENDPOINT_API="http://api:8000/health"

# Define stramlit params
CONTAINER_NAME_STREAMLIT="movie-reco-streamlit-ci"
IMAGE_NAME_STREAMLIT="movie-reco-streamlit:ci"
URL_STREAMLIT="http://localhost:8501"

# Define DB 
DB_URL="sqlite:///./test.db"

# Define docker network
NETWORK_NAME="movie-rco-ci-net"

# Define Summarization vars
API_OK=false
STREAMLIT_OK=false
STREAMLIT_CAN_REACH_API=false

echo "Removing old containers and networks if exists"
docker rm -f $CONTAINER_NAME_API 2>/dev/null
docker rm -f $CONTAINER_NAME_STREAMLIT 2>/dev/null
docker network rm $NETWORK_NAME 2>/dev/null

echo "Building API image"
docker build -f Dockerfile.api -t $IMAGE_NAME_API .
docker build -f Dockerfile.streamlit -t $IMAGE_NAME_STREAMLIT .

echo "Create CI network"
docker network create $NETWORK_NAME

echo "Run containers"
docker run -d \
  --name $CONTAINER_NAME_API \
  --network $NETWORK_NAME \
  --network-alias api \
  -e TESTING=true \
  -p 8000:8000 \
  $IMAGE_NAME_API

docker run -d \
  --name $CONTAINER_NAME_STREAMLIT \
  --network $NETWORK_NAME \
  -e DB_URL="$DB_URL" \
  -e API_URL="http://api:8000" \
  -p 8501:8501 \
  $IMAGE_NAME_STREAMLIT

echo "Waiting for startup..."
sleep 5

echo "Checking if API container is running"
if [ "$(docker inspect -f '{{.State.Running}}' $CONTAINER_NAME_API)" != "true" ]; then
  echo "Container crashed!"
  docker logs $CONTAINER_NAME_API
  exit 1
fi

echo "Logs:"
docker logs $CONTAINER_NAME_API

echo "Testing if api health endpoint is reachable"
for i in {1..10}; do
  if curl -s $HEALTH_ENDPOINT_API; then
    echo -e "\n\nHealth endpoint reachable!\n"
    API_OK=true
    break
  fi
  echo "Waiting..."
  sleep 2
done

echo "Testing if streamlit webserver is reachable"
for i in {1..10}; do
  if curl -s $URL_STREAMLIT; then
    echo -e "\n\nStreamlit web app reachable!\n"
    STREAMLIT_OK=true
    break
  fi
  echo "Waiting for Streamlit web app to start..."
  sleep 2
done


echo "Check if streamlit can reach API health endpoint"
if docker exec "$CONTAINER_NAME_STREAMLIT" \
  python -c "import requests; \
requests.get('$INTERNAL_HEALTH_ENDPOINT_API', timeout=5).raise_for_status()"
then
  echo "Streamlit can reach api health endpoint."
  STREAMLIT_CAN_REACH_API=true
fi

echo "Stop containers and network"
docker stop $CONTAINER_NAME_API
docker stop $CONTAINER_NAME_STREAMLIT
docker network rm $NETWORK_NAME 2>/dev/null

echo -e "\nSummarize results:\n"
if [ "$API_OK" = true ]; then 
  echo "API container run success!"
else 
  echo "API container run failed!"
fi

if [ "$STREAMLIT_OK" = true ]; then
  echo "Streamlit container run success"
else 
  echo "Streamlit container run failed!"
fi

if [ "$STREAMLIT_CAN_REACH_API" == true ]; then
  echo "Streamlit can reach api health endpoint."
else
  echo "Streamlit can't reach api health endpoint."
fi

echo "Done!"