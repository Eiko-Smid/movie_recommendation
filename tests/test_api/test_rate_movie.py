from fastapi import status
from fastapi.testclient import TestClient

from src.api.role import UserRole


# Define Admin and user test data
ADMIN_EMAIL = "admin@test.com"
ADMIN_PWD = "admin"
ADMIN_ID = 1
USER_EMAIL = "user@test.com"
USER_PWD = "user"
USER_ID = 3


#____________________________________________________________________________________________________
# Helpers
#____________________________________________________________________________________________________

def get_admin_auth_head(client: TestClient) -> dict:
    """
    Helper function to obtain admin authentication headers for API requests.
    Logs in as admin and returns the Authorization header with Bearer token.
    """
    response = client.post(
        "/auth/token",
        data={"username": ADMIN_EMAIL, "password": ADMIN_PWD},
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
        "/auth/token",
        data={"username": USER_EMAIL, "password": USER_PWD},
    )

    # Check if token call was successful
    assert response.status_code == status.HTTP_200_OK

    # Extract token and convert to correct format
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}