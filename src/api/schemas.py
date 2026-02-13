from pydantic import BaseModel, EmailStr, Field
from src.api.role import UserRole

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

