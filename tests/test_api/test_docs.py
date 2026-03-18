from fastapi import status
from src.api.main import app


def test_root_endpoint(client):
    '''Test API docs endpoint to ensure api is actually running.'''
    response = client.get("/docs")
    assert response.status_code == status.HTTP_200_OK

    