from fastapi import status
from fastapi.testclient import TestClient
import pytest
import requests

from src.api.role import UserRole
from src.api.main import app
from src.db.database_session import get_db

#____________________________________________________________________________________________________
# Integration tests for /health
#____________________________________________________________________________________________________

def test_health_endpoint_access(client: TestClient, monkeypatch):
    """
    Test that the health endpoint is reachable when DB and MLflow are available.
    Asserts that /health returns 200 OK for a fully healthy system.
    """
    # Set mlflow env var for testing
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://fake-mlflow-server:5000")

    # Simulate mlflow server being up by mocking requests.get to return a successful response
    def mock_requests_get(*args, **kwargs):
        class MockResponse:
            def raise_for_status(self):
                pass  # Simulate a successful response (status code 200)
        return MockResponse()
    
    # Override request with mock request to simulate mlflow server being up
    # Only active during this test function, does not affect other tests
    monkeypatch.setattr("requests.get", mock_requests_get)

    # Call endpoint
    response = client.get(url="/health")

    # Verify response status code
    assert response.status_code == status.HTTP_200_OK


def test_health_endpoint_types_and_vals(client: TestClient, monkeypatch):
    """
    Test that the health endpoint returns the expected types and success values.
    Asserts that the DB and MLflow status payload contains the documented booleans and messages.
    """
    # Set mlflow env var for testing
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://fake-mlflow-server:5000")

    # Simulate mlflow server being up by mocking requests.get to return a successful response
    def mock_requests_get(*args, **kwargs):
        class MockResponse:
            def raise_for_status(self):
                pass  # Simulate a successful response (status code 200)
        return MockResponse()
    
    # Override request with mock request to simulate mlflow server being up
    # Only active during this test function, does not affect other tests
    monkeypatch.setattr("requests.get", mock_requests_get)

    # Call endpoint
    response = client.get(url="/health")

    # Verify response status code
    assert response.status_code == status.HTTP_200_OK

    # Extract data
    data:dict = response.json()
    db_status = data.get("DB")["ok"]
    db_status_msg = data.get("DB")["message"]
    mlflow_status = data.get("MLflow")["ok"]
    mlflow_status_msg = data.get("MLflow")["message"]

    # Verify correct types
    assert isinstance(db_status, bool)
    assert isinstance(db_status_msg, str)
    assert isinstance(mlflow_status, bool)
    assert isinstance(mlflow_status_msg, str)

    # Verify correct vals
    assert db_status is True
    assert db_status_msg == "DB connection healthy."
    assert mlflow_status is True
    assert mlflow_status_msg == "MLflow connection healthy."


@pytest.mark.parametrize(
        "variant, expected_status",
        [
            ("NO DB", status.HTTP_500_INTERNAL_SERVER_ERROR),
            ("NO Tracking URI", status.HTTP_500_INTERNAL_SERVER_ERROR),
            ("No MLflow Server", status.HTTP_500_INTERNAL_SERVER_ERROR),
        ],
)
def test_health_failure(client: TestClient, monkeypatch, variant, expected_status):
    """
    Test that the health endpoint returns 500 for each simulated dependency failure.
    Asserts that DB, tracking URI, and MLflow connectivity failures all mark /health as unhealthy.
    """
    # Set mlflow env var for testing
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://fake-mlflow-server:5000")

    # Simulate mlflow server being up by mocking requests.get to return a successful response
    def mock_requests_get(*args, **kwargs):
        class MockResponse:
            def raise_for_status(self):
                pass  # Simulate a successful response (status code 200)
        return MockResponse()
    
    # Override request with mock request to simulate mlflow server being up
    # Only active during this test function, does not affect other tests
    monkeypatch.setattr("requests.get", mock_requests_get)
    original_get_db_override = app.dependency_overrides.get(get_db)

    if variant == "NO DB":
        # Define fake DB class that raises an exception on execute to simulate DB connection failure
        class DB:
            def execute(self, *args, **kwargs):
                raise Exception("DB connection failed.")
            
        def fake_get_db():
            yield DB()

        # Override get_db dependency with fake DB that simulates connection failure
        app.dependency_overrides[get_db] = fake_get_db

    elif variant == "NO Tracking URI":
        # Delete MLflow tracking uri env var to simulate missing tracking URI
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    elif variant == "No MLflow Server":
        # Simulate mlflow server down 
        def mock_requests_get_failure(*args, **kwargs):
            raise requests.ConnectionError("MLflow server unavailable.")

        monkeypatch.setattr("requests.get", mock_requests_get_failure)

    # Start request
    response = client.get("/health")

    # Check if response status code is 500 Internal Server Error for all failure variants
    assert response.status_code == expected_status

    # Reset get_db override to original state after test
    if original_get_db_override is None:
        app.dependency_overrides.pop(get_db, None)
    else:
        app.dependency_overrides[get_db] = original_get_db_override
