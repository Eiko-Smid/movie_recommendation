import os
import logging
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.api.role import UserRole
from tests.utils.get_authentication_head import (
    get_admin_auth_head,
    get_dev_auth_head,
    get_inactive_admin_auth_head,
    get_user_auth_head,
)

logger = logging.getLogger(__name__)

# ____________________________________________________________________________________________________
# Integration tests for /train/refresh-mv endpoint
# ____________________________________________________________________________________________________

@pytest.mark.parametrize(
    "role, expected_status",
    [
        (UserRole.ADMIN, status.HTTP_200_OK),
        (UserRole.DEVELOPER, status.HTTP_200_OK),
        (UserRole.USER, status.HTTP_403_FORBIDDEN),
    ],
)
def test_refresh_mv_success(client: TestClient, role, expected_status):
    """
    Test that the /train/refresh-mv endpoint successfully logs in.
    Asserts that the response has status code 200 and contains a success message.
    """
    # Get token -> header based on user role
    if role == UserRole.ADMIN:
        header = get_admin_auth_head(client)
    elif role == UserRole.DEVELOPER:
        header = get_dev_auth_head(client)
    elif role == UserRole.USER:
        header = get_user_auth_head(client)

    # Call endpoint to train model
    response = client.post(
        url="/train/refresh-mv",
        headers=header,
    )

    # Check if access is allowed for user
    assert response.status_code == expected_status


def test_refresh_mv_types(client: TestClient):
    """
    Test that the /train/refresh-mv endpoint returns correct field types and values.
    Asserts that the response has correct types for message.
    """
    # Get token -> header
    header = get_admin_auth_head(client)

    # Call endpoint to train model
    response = client.post(
        url="/train/refresh-mv",
        headers=header,
    )

    # Extract data
    data: dict = response.json()

    # Extract data and check types
    assert data.get("status") == "ok"
    assert data.get("concurrent") == True


def test_refresh_mv_inactive_admin(client: TestClient):
    """
    Test that an inactive admin cannot access the /train/refresh-mv endpoint.
    Asserts that the response has status code 401 Unauthorized for an inactive admin user.
    """
    # Get token -> header
    header = get_inactive_admin_auth_head(client)

    # Call endpoint to train model
    response = client.post(
        url="/train/refresh-mv",
        headers=header,
    )

    # Check if access is forbidden for inactive admin user
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


# ____________________________________________________________________________________________________
# Integration tests for /train/train_model endpoint
# ____________________________________________________________________________________________________

@pytest.mark.parametrize(
    "role, expected_status",
    [
        (UserRole.ADMIN, status.HTTP_422_UNPROCESSABLE_CONTENT),
        (UserRole.DEVELOPER, status.HTTP_403_FORBIDDEN),
        (UserRole.USER, status.HTTP_403_FORBIDDEN),
    ],
)
def test_train_model_access_success(client: TestClient, role, expected_status):
    """
    Test role-based access for /train/train_model without request payload.
    For allowed roles, FastAPI returns 422 because the body is required.
    """
    # Get token -> header based on user role
    if role == UserRole.ADMIN:
        header = get_admin_auth_head(client)
    elif role == UserRole.DEVELOPER:
        header = get_dev_auth_head(client)
    elif role == UserRole.USER:
        header = get_user_auth_head(client)

    # No payload on purpose: this should not execute training logic
    response = client.post(
        url="/train/train_model",
        headers=header,
    )

    # Check expected status by role
    assert response.status_code == expected_status


def test_train_model_inactive_admin(client: TestClient):
    """
    Test that an inactive admin cannot access /train/train_model.
    Asserts that the response has status code 401 Unauthorized for an inactive admin user.
    """
    # Get token -> header for inactive admin
    header = get_inactive_admin_auth_head(client)

    # No payload on purpose
    response = client.post(
        url="/train/train_model",
        headers=header,
    )

    # Check if access is forbidden for inactive admin user
    assert response.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.parametrize(
    "service_token, expected_status",
    [
        (os.getenv("API_SERVICE_TOKEN", None), status.HTTP_422_UNPROCESSABLE_CONTENT),
        ("invalid_service_token", status.HTTP_403_FORBIDDEN),
    ]
)
def test_train_model_service_token(client: TestClient, service_token, expected_status):
    
    logging.info(f"Current token is: {service_token}")
    response = client.post(
        url="/train/train_model",
        headers={"api-service-key": service_token},
    )   

    assert response.status_code == expected_status