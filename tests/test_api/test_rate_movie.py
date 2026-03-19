from fastapi import status
from fastapi.testclient import TestClient
import pytest

from src.api.role import UserRole
from src.api.schemas import RateMovieRequest

from tests.utils.get_authentication_head import (
    get_admin_auth_head,
    get_dev_auth_head,
    get_user_auth_head,
    get_inactive_admin_auth_head,
)

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
    """
    Test that users with different roles (admin, developer, user) can rate a movie.
    Asserts that /update_DB/rate_movie returns the expected status for each role.
    """
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
    """
    Test that the rate_movie endpoint returns correct field types and values.
    Asserts that the response has correct types for message, movie_id, user_id, rating, and timestamp.
    """
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
    """
    Test that an inactive admin cannot rate a movie.
    Asserts that /update_DB/rate_movie returns 401 for an inactive admin.
    """
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
    """
    Test that invalid payloads are rejected by the rate_movie endpoint.
    Asserts that /update_DB/rate_movie returns the correct error status for invalid movie_id or rating values.
    """
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

