from fastapi import status
from fastapi.testclient import TestClient
import pytest

from src.api.role import UserRole
from src.api.schemas import RateMovieRequest

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
        "/auth/token",
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
    
    # Check that the request is unauthorized for an inactive admin
    assert response.status_code == status.HTTP_200_OK
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

#____________________________________________________________________________________________________
# Integration tests for /update_DB/rate_movie endpoint
#____________________________________________________________________________________________________

@pytest.mark.parametrize(
    "role, expected_status",
    [
        (UserRole.ADMIN, status.HTTP_200_OK),
        (UserRole.DEVELOPER, status.HTTP_200_OK),
        (UserRole.USER, status.HTTP_200_OK),
    ],
)
def test_rate_movie_user_access(client: TestClient, role, expected_status):
    # Get token -> header
    if role == UserRole.ADMIN:
        header = get_admin_auth_head(client)
    elif role == UserRole.DEVELOPER:
        header = get_dev_auth_head(client)
    elif role == UserRole.USER:
        header = get_user_auth_head(client)
    
    # Define payload
    payload = RateMovieRequest(
        movie_id=1,
        rating=5.0,
    ).model_dump()

    # Call endpoint to rate movie
    response = client.post(
        url="/update_DB/rate_movie",
        json=payload,
        headers=header,
    )

    assert response.status_code == expected_status


def test_rate_movie_types_and_roles(client: TestClient):
    # Get token for admin user
    header = get_admin_auth_head(client)

    # Define payload
    payload = RateMovieRequest(
        movie_id=1,
        rating=5.0,
    ).model_dump()

    # Call endpoint to rate movie
    response = client.post(
        url="/update_DB/rate_movie",
        json=payload,
        headers=header,
    )

    # Extract response data and check types
    data = response.json()
    assert isinstance(data["message"], str)
    assert isinstance(data["movie_id"],int)
    assert isinstance(data["user_id"], int)
    assert isinstance(data["rating"], float)
    assert isinstance(data["timestamp"], int)


def test_rate_movie_inactive_admin(client: TestClient):
    # Get token for inactive admin user
    header = get_inactive_admin_auth_head(client)

    # Define payload
    payload = RateMovieRequest(
        movie_id=1,
        rating=5.0,
    ).model_dump()

    # Call endpoint to rate movie
    response = client.post(
        url="/update_DB/rate_movie",
        json=payload,
        headers=header,
    )

    # Check that the request is unauthorized for an inactive admin
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.parametrize(
    "movie_id, rating, expected_status",
    [
        (10, 3.0, status.HTTP_404_NOT_FOUND),                       # Not existing movie id
        (1, 6.0, status.HTTP_422_UNPROCESSABLE_CONTENT),            # Wrong rating (>5.0 not allowed)
        (1, "wrong_type", status.HTTP_422_UNPROCESSABLE_CONTENT),   # Wrong rating type -> str instead of float
    ],
)
def test_rate_movie_invalid_payload(client: TestClient, movie_id, rating, expected_status):
    # Get token for admin user
    header = get_admin_auth_head(client)

    # Define invalid payload 
    payload = {
        "movie_id": movie_id,
        "rating": rating, 
    }

    # Call endpoint to rate movie
    response = client.post(
        url="/update_DB/rate_movie",
        json=payload,
        headers=header,
    )

    # Check that the response status code matches the expected status for invalid payloads
    assert response.status_code == expected_status

