from fastapi import status
from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    """
    Test API docs endpoint to ensure the API is running and serving documentation.
    Asserts that /docs returns 200 OK.
    """
    response = client.get("/docs")
    assert response.status_code == status.HTTP_200_OK
