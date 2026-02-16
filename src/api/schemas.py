from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Sequence

from src.api.role import UserRole
from src.models.als_movie_rec import ALS_Metrics


class BestParameters(BaseModel):
    """Stores best hyperparameter combination found during training."""
    best_K1: int
    best_B: float
    best_factor: int
    best_reg: float
    best_iters: int


class ALS_Parameter_Grid(BaseModel):
    """Grid parameters for ALS training (used by the `/train` endpoint)."""
    bm25_K1_list: Sequence[int] = Field((100, 200), description="BM25 K1 values")
    bm25_B_list: Sequence[float] = Field((0.8, 1.0), description="BM25 B values")
    factors_list: Sequence[int] = Field((128, 256), description="latent factors")
    reg_list: Sequence[float] = Field((0.10, 0.20), description="regularization")
    iters_list: Sequence[int] = Field((25,), description="ALS iterations")

#___________________________________________________________________________________________________
# General schemas
#___________________________________________________________________________________________________




#___________________________________________________________________________________________________
# Request schemas
#___________________________________________________________________________________________________

class UserRoleRequest(BaseModel):
    """Request model for changing a user's role.

    Fields:
    - role: the target role to assign to the user.
    """
    role: UserRole = Field(..., description="Target role to assign (USER, DEVELOPER, ADMIN)")

class ActiveUserRequest(BaseModel):
    """Request model to activate or deactivate a user.

    Fields:
    - is_active: True to activate, False to deactivate.
    """
    is_active: bool = Field(..., description="True to activate the user, False to deactivate")

class UserCreate(BaseModel):
    """Request model used when creating a new user.

    Fields:
    - email: user's email address
    - password: plain-text password (will be hashed before storage)
    """
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="Plain-text password (will be hashed)")


class TrainRequest(BaseModel):
    """Input schema for `/train`."""
    n_users: int = Field(1000, description="Number of users to read (0 = full dataset)")
    pos_threshold: float = Field(4.0, description="Threshold for positive rating")
    als_parameter: ALS_Parameter_Grid = Field(..., description="ALS hyperparameter grid")
    n_popular_movies: int = Field(100, description="# popular movies for cold-start")


class RecommendRequest(BaseModel):
    """Input schema for the `/recommend` endpoint."""
    user_id: int = Field(
        ...,
        description="ID of the user for whom to generate recommendations.",
        example=42,
    )
    n_movies_to_rec: int = Field(
        5,
        gt=0,
        le=100,
        description="Number of movie recommendations to return (1–100).",
        example=10,
    )
    new_user_interactions: Optional[List[int]] = Field(
        None,
        description=(
            "Optional list of movie IDs recently watched or liked by the user. "
            "Used for cold-start or fold-in recommendations."
        ),
        example=[296, 318, 593],
    )


#___________________________________________________________________________________________________
# Response schemas
#___________________________________________________________________________________________________

class Token(BaseModel):
    """Authentication token response.

    Fields:
    - access_token: the JWT access token
    - token_type: token type (usually "bearer")
    """
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type (usually 'bearer')")


class ProtectedResponse(BaseModel):
    """Response model for protected endpoints used in examples/tests.

    Fields:
    - message: informational message
    - user_email: email of the authenticated user
    """
    message: str = Field(..., description="Informational message")
    user_email: str = Field(..., description="Authenticated user's email")


class UserResponse(BaseModel):
    """Admin-facing user representation returned by admin endpoints.

    Fields:
    - id: numeric user identifier
    - email: user's email address
    - is_active: whether the account is active
    - role: assigned role for the user
    """
    id: int = Field(..., description="User ID")
    email: EmailStr = Field(..., description="User email address")
    is_active: bool = Field(..., description="Whether the user account is active")
    role: UserRole = Field(..., description="Assigned user role")


class GetAllUsersResponse(BaseModel):
    """Response model returning a list of all users for admin listing endpoints.

    Fields:
    - users: list of `UserAdminResponse` objects
    """
    users: list[UserResponse] = Field(..., description="List of users")


class TrainResponse(BaseModel):
    """Output schema for `/train`."""
    best_param: Optional[BestParameters] = Field(None, description="Best hyperparameters")
    best_metrics: Optional[ALS_Metrics] = Field(None, description="Best evaluation metrics")


class RecommendResponse(BaseModel):
    """Output schema for the `/recommend` endpoint."""
    user_id: int = Field(..., description="User ID for which recommendations were generated.")
    movie_ids: List[int] = Field(..., description="List of recommended movie IDs sorted by relevance.")
    movie_titles: List[str] = Field(..., description="List of corresponding movie titles.")
    movie_genres: List[str] = Field(..., description="List of corresponding movie genres.")