from fastapi import status
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    '''Test API docs endpoint to ensure api is actually running.'''
    response = client.get("/docs")
    assert response.status_code == status.HTTP_200_OK