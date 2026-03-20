from fastapi import status
from fastapi.testclient import TestClient


def test_register_success(client: TestClient):
    """
    Test successful user registration with a valid payload.
    Asserts that the /auth/register endpoint returns 201 and the correct success message.
    """
    payload = {"email": "user@example.com", "password": "string"}

    # Call registration endpoint
    response = client.post("/auth/register", json=payload)

    # Check status
    assert response.status_code == status.HTTP_201_CREATED

    # Check if request was successfull
    data = response.json()
    assert data["message"] == "Created user successfully."


def test_register_invalid_payload(client: TestClient):
    """
    Test registration with an invalid payload (wrong email type).
    Asserts that the /auth/register endpoint returns 422 for invalid input.
    """
    payload = {"email": 123, "password": "string"}

    # Call registration endpoint with wrong email type
    response = client.post("/auth/register", json=payload)

    # Check status
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_register_duplicated_email(client: TestClient):
    """
    Test registration with a duplicated email.
    Asserts that the /auth/register endpoint returns 400 and the correct error message.
    """
    payload = {"email": "user@example.com", "password": "string"}

    # Call registration endpoint
    response = client.post("/auth/register", json=payload)

    # Check status
    assert response.status_code == status.HTTP_400_BAD_REQUEST

    # Check if request failed
    data = response.json()
    assert data["detail"] == "Email already exists."


def test_login_success(client: TestClient):
    """
    Test successful login with valid admin credentials.
    Asserts that /auth/token returns 200 and a valid access token.
    """
    response = client.post(
        "/auth/token", data={"username": "admin@test.com", "password": "admin"}
    )
    assert response.status_code == 200

    # Check response
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_wrong_password(client: TestClient):
    """
    Test login with a wrong password for an existing user.
    Asserts that /auth/token returns 401 and the correct error message.
    """
    response = client.post(
        "/auth/token", data={"username": "admin@test.com", "password": "wrong_password"}
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED

    # Check response
    data = response.json()
    assert data["detail"] == "Bad credentials."
