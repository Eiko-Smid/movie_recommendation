import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.api.role import UserRole
from src.api.schemas import (
    RecommendMovieByIDRequest,
    RecommendMovieCurrentUserRequest,
)
from tests.utils.get_authentication_head import (
    get_admin_auth_head,
    get_dev_auth_head,
    get_user_auth_head,
)

# ____________________________________________________________________________________________________
# Integration tests for /recommend/recommend_movie_for_current_user endpoint
# ____________________________________________________________________________________________________


@pytest.mark.parametrize(
    "role, expected_status",
    [
        (UserRole.ADMIN, status.HTTP_200_OK),
        (UserRole.DEVELOPER, status.HTTP_200_OK),
        (UserRole.USER, status.HTTP_200_OK),
    ],
)
def test_recommend_movie_for_current_user_access(
    client: TestClient, role: UserRole, expected_status: int
) -> None:
    """
    Test that users with different roles (admin, developer, user) can access the
    /recommend/recommend_movie_for_current_user endpoint. Asserts that the endpoint
    returns the expected status code for each role.
    """
    # Get token -> header based on user role
    if role == UserRole.ADMIN:
        header = get_admin_auth_head(client)
    elif role == UserRole.DEVELOPER:
        header = get_dev_auth_head(client)
    elif role == UserRole.USER:
        header = get_user_auth_head(client)

    # Define payload for recommendation request
    payload = RecommendMovieCurrentUserRequest(
        n_movies_to_rec=10,
        new_user_interactions=[1, 2, 3],
    ).model_dump()

    # Call endpoint to get movie recommendations for current user
    response = client.post(
        url="/recommend/recommend_movie_for_current_user",
        json=payload,
        headers=header,
    )

    # Check if access is allowed for the user
    assert response.status_code == expected_status


def test_recommend_movie_for_current_user_types_and_roles(client: TestClient) -> None:
    """
    Test that the /recommend/recommend_movie_for_current_user endpoint returns the
    correct response types. Checks that the response contains a string message and
    a list of recommended_movie_ids.
    """
    # Get header for user role
    header = get_user_auth_head(client)

    # Define payload for recommendation request
    payload = RecommendMovieCurrentUserRequest(
        n_movies_to_rec=10,
        new_user_interactions=[1, 2, 3],
    ).model_dump()

    # Call endpoint to get movie recommendations for current user
    response = client.post(
        url="/recommend/recommend_movie_for_current_user",
        json=payload,
        headers=header,
    )

    # Extract response JSON and check types
    data: dict = response.json()
    assert isinstance(data.get("user_id"), int)
    assert all(isinstance(movie_id, int) for movie_id in data.get("movie_ids"))
    assert all(isinstance(title, str) for title in data.get("movie_titles", []))
    assert all(isinstance(genre, str) for genre in data.get("movie_genres", []))


def test_recommend_movie_for_current_user_champion_model_not_loaded(
    client: TestClient,
) -> None:
    """
    Test that the /recommend/recommend_movie_for_current_user endpoint returns a
    503 Service Unavailable error when the champion model is not loaded.
    """
    # Override champion model dependency to simulate champion model not loaded
    client.app.state.app_state.champ_model = None

    # Get header for user role
    header = get_user_auth_head(client)

    # Define payload for recommendation request
    payload = RecommendMovieCurrentUserRequest(
        n_movies_to_rec=10,
        new_user_interactions=[1, 2, 3],
    ).model_dump()

    # Call endpoint to get movie recommendations for current user
    response = client.post(
        url="/recommend/recommend_movie_for_current_user",
        json=payload,
        headers=header,
    )

    # Check that response status code is 503 Service Unavailable when champion model is not loaded
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


@pytest.mark.parametrize(
    "n_movies_to_rec, new_user_interactions, expected_status",
    [
        (
            0,
            [1, 2, 3],
            status.HTTP_422_UNPROCESSABLE_CONTENT,
        ),  # Invalid n_movies_to_rec
        (
            10,
            "invalid_interactions",
            status.HTTP_422_UNPROCESSABLE_CONTENT,
        ),  # Invalid new_user_interactions
        (
            10,
            None,
            status.HTTP_200_OK,
        ),  # Valid payload with optional new_user_interactions
    ],
)
def test_recommend_movie_for_current_user_invalid_payload(
    client: TestClient,
    n_movies_to_rec: int,
    new_user_interactions,
    expected_status: int,
) -> None:
    """
    Test that the /recommend/recommend_movie_for_current_user endpoint returns a
    422 Unprocessable Entity error when the request payload is invalid.
    """
    # Get header for user role
    header = get_user_auth_head(client)

    # Define payload for recommendation request
    payload = {
        "n_movies_to_rec": n_movies_to_rec,
        "new_user_interactions": new_user_interactions,
    }
    # Call endpoint to get movie recommendations for current user with invalid payload
    response = client.post(
        url="/recommend/recommend_movie_for_current_user",
        json=payload,
        headers=header,
    )

    # Check that response status code is 422 Unprocessable Entity when payload is invalid
    assert response.status_code == expected_status


# ____________________________________________________________________________________________________
# Integration tests for /recommend/recommend_movie_by_id endpoint
# ____________________________________________________________________________________________________


@pytest.mark.parametrize(
    "role, expected_status",
    [
        (UserRole.ADMIN, status.HTTP_200_OK),
        (UserRole.DEVELOPER, status.HTTP_200_OK),
        (UserRole.USER, status.HTTP_403_FORBIDDEN),
    ],
)
def test_recommend_movie_by_id_access(
    client: TestClient,
    role: UserRole,
    expected_status: int,
) -> None:
    """
    Test that only admin and developer users can access the
    /recommend/recommend_movie_by_id endpoint.
    """
    # Get token -> header based on user role
    if role == UserRole.ADMIN:
        header = get_admin_auth_head(client)
    elif role == UserRole.DEVELOPER:
        header = get_dev_auth_head(client)
    elif role == UserRole.USER:
        header = get_user_auth_head(client)

    # Define payload for recommendation request
    payload = RecommendMovieByIDRequest(
        user_id=1,
        n_movies_to_rec=10,
        new_user_interactions=[1, 2, 3],
    ).model_dump()

    # Call endpoint to get movie recommendations by user id
    response = client.post(
        url="/recommend/recommend_movie_by_id",
        json=payload,
        headers=header,
    )

    # Check if access is allowed for the user role
    assert response.status_code == expected_status


def test_recommend_movie_by_id_types_and_roles(client: TestClient) -> None:
    """
    Test that the /recommend/recommend_movie_by_id endpoint returns
    the expected response field types.
    """
    # Get header for an allowed role
    header = get_admin_auth_head(client)

    # Define payload for recommendation request
    payload = RecommendMovieByIDRequest(
        user_id=1,
        n_movies_to_rec=10,
        new_user_interactions=[1, 2, 3],
    ).model_dump()

    # Call endpoint to get movie recommendations by user id
    response = client.post(
        url="/recommend/recommend_movie_by_id",
        json=payload,
        headers=header,
    )

    # Extract response JSON and check types
    data: dict = response.json()
    assert isinstance(data.get("user_id"), int)
    assert all(isinstance(movie_id, int) for movie_id in data.get("movie_ids", []))
    assert all(isinstance(title, str) for title in data.get("movie_titles", []))
    assert all(isinstance(genre, str) for genre in data.get("movie_genres", []))


def test_recommend_movie_by_id_champion_model_not_loaded(client: TestClient) -> None:
    """
    Test that the /recommend/recommend_movie_by_id endpoint returns a
    503 Service Unavailable error when the champion model is not loaded.
    """
    # Override champion model dependency to simulate champion model not loaded
    client.app.state.app_state.champ_model = None

    # Get header for an allowed role
    header = get_admin_auth_head(client)

    # Define payload for recommendation request
    payload = RecommendMovieByIDRequest(
        user_id=1,
        n_movies_to_rec=10,
        new_user_interactions=[1, 2, 3],
    ).model_dump()

    # Call endpoint to get movie recommendations by user id
    response = client.post(
        url="/recommend/recommend_movie_by_id",
        json=payload,
        headers=header,
    )

    # Check that response status code is 503 if champion model is unavailable
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


@pytest.mark.parametrize(
    "user_id, n_movies_to_rec, new_user_interactions, expected_status",
    [
        (1, 0, [1, 2, 3], status.HTTP_422_UNPROCESSABLE_CONTENT),
        ("invalid_user_id", 10, [1, 2, 3], status.HTTP_422_UNPROCESSABLE_CONTENT),
        (1, 10, "invalid_interactions", status.HTTP_422_UNPROCESSABLE_CONTENT),
        (1, 10, None, status.HTTP_200_OK),
    ],
)
def test_recommend_movie_by_id_invalid_payload(
    client: TestClient,
    user_id,
    n_movies_to_rec: int,
    new_user_interactions,
    expected_status: int,
) -> None:
    """
    Test that the /recommend/recommend_movie_by_id endpoint validates input
    and returns 422 for invalid payloads.
    """
    # Get header for an allowed role
    header = get_admin_auth_head(client)

    # Use raw JSON so FastAPI/Pydantic validates invalid cases at request time
    payload = {
        "user_id": user_id,
        "n_movies_to_rec": n_movies_to_rec,
        "new_user_interactions": new_user_interactions,
    }

    # Call endpoint with payload variations
    response = client.post(
        url="/recommend/recommend_movie_by_id",
        json=payload,
        headers=header,
    )

    # Check expected status for each payload scenario
    assert response.status_code == expected_status
