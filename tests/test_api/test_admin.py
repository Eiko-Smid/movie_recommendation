import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.api.role import UserRole
from src.api.schemas import ActiveUserRequest, UserRoleRequest
from tests.utils.get_authentication_head import (
    USER_ID,
    get_admin_auth_head,
    get_dev_auth_head,
    get_inactive_admin_auth_head,
    get_user_auth_head,
)

# ____________________________________________________________________________________________________
# Integration tests for /get_all_users endpoint
# ____________________________________________________________________________________________________


@pytest.mark.parametrize(
    "role, expected_status",
    [
        (UserRole.ADMIN, status.HTTP_200_OK),
        (UserRole.DEVELOPER, status.HTTP_403_FORBIDDEN),
        (UserRole.USER, status.HTTP_403_FORBIDDEN),
    ],
)
def test_get_all_users_access(client: TestClient, role, expected_status) -> None:
    """
    Test that a non-admin user cannot access the get_all_users endpoint.
    Asserts that /admin/get_all_users returns 403 Forbidden for regular users.
    """
    # Get token
    if role == UserRole.ADMIN:
        headers = get_admin_auth_head(client)
    elif role == UserRole.DEVELOPER:
        headers = get_dev_auth_head(client)
    elif role == UserRole.USER:
        headers = get_user_auth_head(client)

    # call endpoint with corresponding authorization
    response = client.post(
        url="/admin/get_all_users",
        headers=headers,
    )

    # Check if access is forbidden for the user
    assert response.status_code == expected_status


def test_get_all_users_types_and_roles(client: TestClient) -> None:
    """
    Test that all returned users have correct field types and roles.
    Asserts that each user dict has the correct types for id, email, is_active, and role.
    """
    # Get admin token
    headers = get_admin_auth_head(client)

    # Call endpoint with corresponding authorization
    response = client.post(
        url="/admin/get_all_users",
        headers=headers,
    )

    # Extract data
    users = response.json()["users"]

    # Check types and roles
    assert all(isinstance(user["id"], int) for user in users)
    assert all(isinstance(user["email"], str) for user in users)
    assert all(isinstance(user["is_active"], bool) for user in users)
    assert all(isinstance(user["role"], str) for user in users)


def test_get_all_users_by_inactive_admin(client: TestClient) -> None:
    """
    Test that an inactive admin cannot access the get_all_users endpoint.
    Asserts that /admin/get_all_users returns 401 for an inactive admin.
    """
    # Get token for inactive admin
    headers = get_inactive_admin_auth_head(client)

    # Call endpoint with corresponding authorization
    response = client.post(
        url="/admin/get_all_users",
        headers=headers,
    )

    # Check that the request is unauthorized for an inactive admin
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


# ____________________________________________________________________________________________________
# Integration tests for "/users/{user_id}/role" endpoint
# ____________________________________________________________________________________________________


@pytest.mark.parametrize(
    "role, expected_status",
    [
        (UserRole.ADMIN, status.HTTP_200_OK),
        (UserRole.DEVELOPER, status.HTTP_403_FORBIDDEN),
        (UserRole.USER, status.HTTP_403_FORBIDDEN),
    ],
)
def test_set_user_role_access(client: TestClient, role, expected_status) -> None:
    """
    Test that an admin can successfully change a user's role.
    Asserts that /admin/users/{user_id}/role returns 200 and updates the user's role correctly.
    """
    # Get token
    if role == UserRole.ADMIN:
        headers = get_admin_auth_head(client)
    elif role == UserRole.DEVELOPER:
        headers = get_dev_auth_head(client)
    elif role == UserRole.USER:
        headers = get_user_auth_head(client)

    # Define payload
    payload = UserRoleRequest(role=UserRole.DEVELOPER).model_dump()

    # Call endpoint to change user role to developer
    response = client.patch(
        url=f"/admin/users/{USER_ID}/role",
        json=payload,
        headers=headers,
    )

    # Validate successful call
    assert response.status_code == expected_status


def test_set_user_role_types_and_roles(client: TestClient) -> None:
    """
    Test that the set_user_role endpoint returns correct field types and updated role.
    Asserts that the response has correct types for id, email, is_active, and role after update.
    """
    # Get admin token and call endpoint to change role
    headers = get_admin_auth_head(client)

    # Define payload
    payload = UserRoleRequest(role=UserRole.USER).model_dump()

    # Call endpoint to change user role to user
    response = client.patch(
        url=f"/admin/users/{USER_ID}/role",
        json=payload,
        headers=headers,
    )

    # Extract data and check types
    data = response.json()
    assert isinstance(data["id"], int)
    assert isinstance(data["email"], str)
    assert isinstance(data["is_active"], bool)
    assert isinstance(data["role"], str)


def test_set_user_role_by_inactive_admin(client: TestClient) -> None:
    """
    Test that an inactive admin cannot change a user's role.
    Asserts that /admin/users/{user_id}/role returns 401 for an inactive admin.
    """
    # Get token for inactive admin
    headers = get_inactive_admin_auth_head(client)

    # Define payload
    payload = UserRoleRequest(role=UserRole.DEVELOPER).model_dump()

    # Call endpoint to change user role to developer
    response = client.patch(
        url=f"/admin/users/{USER_ID}/role",
        json=payload,
        headers=headers,
    )
    # Check that the request is unauthorized for an inactive admin
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


# ____________________________________________________________________________________________________
# Integration tests for "/users/{user_id}/is_active" endpoint
# ____________________________________________________________________________________________________


@pytest.mark.parametrize(
    "role, expected_status",
    [
        (UserRole.ADMIN, status.HTTP_200_OK),
        (UserRole.DEVELOPER, status.HTTP_403_FORBIDDEN),
        (UserRole.USER, status.HTTP_403_FORBIDDEN),
    ],
)
def test_set_is_active_access(client: TestClient, role, expected_status) -> None:
    """
    Test that an admin can successfully change a user's role.
    Asserts that /admin/users/{user_id}/role returns 200 and updates the user's role correctly.
    """
    # Get token
    if role == UserRole.ADMIN:
        headers = get_admin_auth_head(client)
    elif role == UserRole.DEVELOPER:
        headers = get_dev_auth_head(client)
    elif role == UserRole.USER:
        headers = get_user_auth_head(client)

    # Define payload
    payload = ActiveUserRequest(is_active=True).model_dump()

    # Call endpoint to update active state of user with id "USER_ID"
    response = client.patch(
        url=f"/admin/users/{USER_ID}/is_active",
        json=payload,
        headers=headers,
    )

    # Validate successful call
    assert response.status_code == expected_status


def test_set_user_active_types_and_roles(client: TestClient) -> None:
    """
    Test that the set_user_active endpoint returns correct field types and updated is_active flag.
    Asserts that the response has correct types for id, email, is_active, and role after update.
    """
    # Get admin token and call endpoint to change active status
    headers = get_admin_auth_head(client)

    # Define payload
    payload = ActiveUserRequest(is_active=True).model_dump()

    # Call endpoint to change user active status to true
    response = client.patch(
        url=f"/admin/users/{USER_ID}/is_active",
        json=payload,
        headers=headers,
    )

    # Extract data and check types
    data = response.json()
    assert isinstance(data["id"], int)
    assert isinstance(data["email"], str)
    assert isinstance(data["is_active"], bool)
    assert isinstance(data["role"], str)


def test_set_user_active_by_inactive_admin(client: TestClient) -> None:
    """
    Test that an inactive admin cannot change a user's active status.
    Asserts that /admin/users/{user_id}/is_active returns 401 for an inactive admin.
    """
    # Get token for inactive admin
    headers = get_inactive_admin_auth_head(client)

    # Define payload
    payload = ActiveUserRequest(is_active=True).model_dump()

    # Call endpoint to change user active status to true
    response = client.patch(
        url=f"/admin/users/{USER_ID}/is_active",
        json=payload,
        headers=headers,
    )

    # Check that the request is unauthorized for an inactive admin
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
