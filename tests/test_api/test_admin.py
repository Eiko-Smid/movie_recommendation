from fastapi import status
from fastapi.testclient import TestClient

from src.api.role import UserRole


# Define Admin and user test data
ADMIN_EMAIL = "admin@test.com"
ADMIN_PWD = "admin"
USER_EMAIL = "user@test.com"
USER_PWD = "user"


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


def test_get_all_users_success(client: TestClient) -> None:
    """
    Test that an admin can successfully retrieve all users.
    Asserts that /admin/get_all_users returns 200 and a valid users list.
    """
    # Call endpoint
    headers = get_admin_auth_head(client)
    response = client.post(
        "/admin/get_all_users",
        headers=headers,
    )
    # Validate successful call
    assert response.status_code == status.HTTP_200_OK
    
    # Check data type of response
    data = response.json()
    assert "users" in data
    assert isinstance(data["users"], list)
    assert len(data["users"]) >= 1
    

def test_get_all_users_types_and_roles(client: TestClient) -> None:
    """
    Test that all returned users have correct field types and roles.
    Asserts that each user dict has the correct types for id, email, is_active, and role.
    """
    # Call endpoint
    headers = get_admin_auth_head(client)
    response = client.post(
        "/admin/get_all_users",
        headers=headers,
    )

    # Extract data
    users = response.json()["users"]
    
    # Check types and roles
    assert all(isinstance(user["id"], int) for user in users)
    assert all(isinstance(user["email"], str) for user in users)
    assert all(isinstance(user["is_active"], bool) for user in users)
    assert all(isinstance(user["role"], str) for user in users)


def test_get_all_users_forbidden(client: TestClient) -> None:
    """
    Test that a non-admin user cannot access the get_all_users endpoint.
    Asserts that /admin/get_all_users returns 403 Forbidden for regular users.
    """
    headers = get_user_auth_head(client)
    response = client.post(
        "/admin/get_all_users",
        headers=headers,
    )

    # Check if access is forbidden for the user
    assert response.status_code == status.HTTP_403_FORBIDDEN
    