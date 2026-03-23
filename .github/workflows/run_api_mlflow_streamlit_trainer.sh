#!/bin/bash

############################################################
# CI LOCAL SIMULATION SCRIPT
# ----------------------------------------------------------
# This script replicates the CI pipeline locally:
# 1) Build Docker images
# 2) Run containers
# 3) Test containers (health + networking)
# 4) Cleanup everything
#
# Variables are identical to ci.yml
############################################################

set -e  # Stop on error

#########################
# ENV VARIABLES (same as CI)
#########################

# API
IMAGE_NAME_API="movie-reco-api:latest"
CONTAINER_NAME_API="movie-reco-api-ci"
HEALTH_ENDPOINT_API="http://localhost:8000/health"
API_SERVICE_TOKEN="test-service-token"

# Streamlit
IMAGE_NAME_STREAMLIT="movie-reco-streamlit:latest"
CONTAINER_NAME_STREAMLIT="movie-reco-streamlit-ci"
URL_STREAMLIT="http://localhost:8501"
INTERNAL_HEALTH_ENDPOINT_API="http://api:8000/health"

# Trainer
IMAGE_NAME_TRAINER="movie-reco-trainer:latest"
CONTAINER_NAME_TRAINER="movie-reco-trainer-ci"

# MLflow
IMAGE_NAME_MLFLOW="movie-reco-mlflow:latest"
CONTAINER_NAME_MLFLOW="movie-reco-mlflow-ci"
URL_MLFLOW="http://localhost:5000"

# DB
DB_URL="sqlite:///./test.db"

# Network
NETWORK_NAME="movie-rco-ci-net"

# Testing
TESTING="true"

#########################
# BUILD IMAGES
#########################

echo "Building Docker images..."

docker build -f Dockerfile.api -t $IMAGE_NAME_API .
docker build -f Dockerfile.streamlit -t $IMAGE_NAME_STREAMLIT .
docker build -f Dockerfile.trainer -t $IMAGE_NAME_TRAINER .
docker build -f Dockerfile.mlflow -t $IMAGE_NAME_MLFLOW .

#########################
# CREATE NETWORK
#########################

echo "Creating network..."
docker network rm $NETWORK_NAME 2>/dev/null || true
docker network create $NETWORK_NAME

#########################
# RUN CONTAINERS
#########################

echo "Starting MLflow..."
docker rm -f $CONTAINER_NAME_MLFLOW 2>/dev/null || true
docker run -d \
  --name $CONTAINER_NAME_MLFLOW \
  --network $NETWORK_NAME \
  -p 5000:5000 \
  $IMAGE_NAME_MLFLOW

echo "Starting API..."
docker rm -f $CONTAINER_NAME_API 2>/dev/null || true
docker run -d \
  --name $CONTAINER_NAME_API \
  --network $NETWORK_NAME \
  --network-alias api \
  -e TESTING=true \
  -e MLFLOW_TRACKING_URI=http://$CONTAINER_NAME_MLFLOW:5000 \
  -p 8000:8000 \
  $IMAGE_NAME_API

echo "Starting Streamlit..."
docker rm -f $CONTAINER_NAME_STREAMLIT 2>/dev/null || true
docker run -d \
  --name $CONTAINER_NAME_STREAMLIT \
  --network $NETWORK_NAME \
  -e DB_URL="$DB_URL" \
  -e API_URL="http://api:8000" \
  -p 8501:8501 \
  $IMAGE_NAME_STREAMLIT

echo "Starting Trainer..."
docker rm -f $CONTAINER_NAME_TRAINER 2>/dev/null || true
docker run -d \
  --name $CONTAINER_NAME_TRAINER \
  --network $NETWORK_NAME \
  -e DB_URL="$DB_URL" \
  -e API_URL="http://api:8000" \
  $IMAGE_NAME_TRAINER

#########################
# TEST MLflow
#########################

echo "Testing MLflow..."
for i in {1..10}; do
  if curl -s $URL_MLFLOW > /dev/null; then
    echo "MLflow reachable"
    break
  fi
  echo "Waiting for MLflow..."
  sleep 2
done

#########################
# TEST API
#########################

echo "Testing API..."
for i in {1..10}; do
  if curl -s $HEALTH_ENDPOINT_API > /dev/null; then
    echo "API reachable"
    break
  fi
  echo "Waiting for API..."
  sleep 2
done

#########################
# TEST STREAMLIT
#########################

echo "Testing Streamlit..."
for i in {1..10}; do
  if curl -s $URL_STREAMLIT > /dev/null; then
    echo "Streamlit reachable"
    break
  fi
  echo "Waiting for Streamlit..."
  sleep 2
done

#########################
# TEST TRAINER → API
#########################

echo "Testing Trainer → API..."
for i in {1..10}; do
  if docker exec "$CONTAINER_NAME_TRAINER" curl -s $INTERNAL_HEALTH_ENDPOINT_API > /dev/null; then
    echo "Trainer can reach API"
    break
  fi
  echo "Waiting for Trainer..."
  sleep 2
done

#########################
# TEST STREAMLIT → API
#########################

echo "Testing Streamlit → API..."
if docker exec "$CONTAINER_NAME_STREAMLIT" \
  python -c "import requests; requests.get('$INTERNAL_HEALTH_ENDPOINT_API', timeout=5).raise_for_status()"
then
  echo "Streamlit can reach API"
else
  echo "Streamlit cannot reach API"
  docker logs $CONTAINER_NAME_API
  docker logs $CONTAINER_NAME_STREAMLIT
  exit 1
fi

#########################
# CLEANUP
#########################

echo "Cleaning up..."

docker stop $CONTAINER_NAME_API 2>/dev/null || true
docker stop $CONTAINER_NAME_STREAMLIT 2>/dev/null || true
docker stop $CONTAINER_NAME_TRAINER 2>/dev/null || true
docker stop $CONTAINER_NAME_MLFLOW 2>/dev/null || true

docker rm -f $CONTAINER_NAME_API $CONTAINER_NAME_STREAMLIT $CONTAINER_NAME_TRAINER $CONTAINER_NAME_MLFLOW 2>/dev/null || true

docker network rm $NETWORK_NAME 2>/dev/null || true

echo "CI simulation finished successfully"