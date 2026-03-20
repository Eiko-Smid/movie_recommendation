from fastapi import status
from fastapi.testclient import TestClient

# Define Admin and user test data
ADMIN_EMAIL = "admin@test.com"
ADMIN_PWD = "admin"
ADMIN_ID = 1

DEV_EMAIL = "dev@test.com"
DEV_PWD = "dev"
DEV_ID = 2

USER_EMAIL = "user@test.com"
USER_PWD = "user"
USER_ID = 3

# Define inactive admin test data
INACTIVE_ADMIN_EMAIL = "inactive_admin@test.com"
INACTIVE_ADMIN_PWD = "inactiveadmin"
INACTIVE_ADMIN_ID = 99


def get_admin_auth_head(client: TestClient) -> dict:
    """
    Helper function to obtain admin authentication headers for API requests.
    Logs in as admin and returns the Authorization header with Bearer token.
    """
    response = client.post(
        url="/auth/token",
        data={"username": ADMIN_EMAIL, "password": ADMIN_PWD},
    )

    # Check if token call was successful
    assert response.status_code == status.HTTP_200_OK

    # Extract token and convert to correct format
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def get_dev_auth_head(client: TestClient) -> dict:
    """
    Helper function to obtain developer authentication headers for API requests.
    Logs in as developer and returns the Authorization header with Bearer token.
    """
    response = client.post(
        url="/auth/token",
        data={"username": DEV_EMAIL, "password": DEV_PWD},
    )

    # Check if token call was successful
    assert response.status_code == status.HTTP_200_OK

    # Extract token and convert to correct format
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def get_user_auth_head(client: TestClient) -> dict:
    """
    Helper function to obtain regular user authentication headers for API requests.
    Logs in as a normal user and returns the Authorization header with Bearer token.
    """
    response = client.post(
        url="/auth/token",
        data={"username": USER_EMAIL, "password": USER_PWD},
    )

    # Check if token call was successful
    assert response.status_code == status.HTTP_200_OK

    # Extract token and convert to correct format
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def get_inactive_admin_auth_head(client: TestClient) -> dict:
    """
    Helper function to obtain inactive admin authentication headers for API requests.
    Logs in as inactive admin and returns the Authorization header with Bearer token.
    """
    response = client.post(
        "/auth/token",
        data={"username": INACTIVE_ADMIN_EMAIL, "password": INACTIVE_ADMIN_PWD},
    )
    # Should still get a token if login is allowed, but endpoints should reject
    assert response.status_code == status.HTTP_200_OK
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
