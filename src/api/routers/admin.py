from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from sqlalchemy.orm import Session

from src.db.database_session import get_db
from src.db.users import User
from src.api.security import get_current_user, check_user_authorization
from src.api.role import UserRole
from src.api.schemas import UserResponse, GetAllUsersResponse, UserRoleRequest, ActiveUserRequest


# Define router
router = APIRouter(prefix="/admin", tags=["admin"])


@router.post(
        "/get_all_users",
        response_model=GetAllUsersResponse
)
def get_all_users(
    db: Session = Depends(get_db),
    _: User = Depends(check_user_authorization(UserRole.ADMIN))
):
    """
    Returns a list of all users (id, email, is_active, role) in the DB, when the called
    user has admin rights.

    **Returns**:\n
    `GetAllUsersResponse`(response_model):
        users (list of UserResponse): A list of all users in the DB with their basic fields

    `Requires:` Admin privileges
    """
    # Search for all users inside DB
    all_users = db.query(User).order_by(User.id.asc()).all()
    
    return GetAllUsersResponse(
        users= [
            UserResponse(
                id=user.id,
                email=user.email,
                is_active=user.is_active,
                role=user.role,
            ) for user in all_users
        ]
    )


@router.patch(
    "/users/{user_id}/role",
    response_model=UserResponse,
    tags=["admin"],
)
def set_user_role(
    user_id: int,
    user_role: UserRoleRequest,
    db: Session = Depends(get_db),
    _: User = Depends(check_user_authorization(UserRole.ADMIN)),
):
    """
    Sets the role of a specific user.

    **Parameters:**\n
    `user_id` (int, path): The unique ID of the user whose role should be updated\n
    `user_role` (UserRoleRequest, body): The new role to assign to the user (USER, DEVELOPER, or ADMIN)\n

    **Returns:**\n
    `UserResponse`(response_model):
    - `id` (int): The user ID
    - `email` (str): The user's email address
    - `is_active` (bool): Whether the user account is active
    - `role` (str): The user's new role

    **Requires:** Admin privileges
    """
    # Check if the user that correspondes to the given user_id exists in DB
    user = db.query(User).filter(User.id == user_id).first()

    # Raise error if user doesn't exist
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found."
        )
    
    # Update user role 
    user.role = user_role.role
    db.commit()
    db.refresh(user)

    return UserResponse(
        id=user.id,
        email=user.email,
        is_active=user.is_active,
        role=user.role
    )



@router.patch(
    "/users/{user_id}/is_active",  # ✅ leading slash
    response_model=UserResponse,
    tags=["admin"],
)
def change_active_state(
user_id: int,
    active_request: ActiveUserRequest,
    db: Session = Depends(get_db),
    _ : User = Depends(check_user_authorization(UserRole.ADMIN)),
):
    '''
    Update the is_active flag of a specific user.
    
    **Parameters:**\n
    `user_id` (int, path): The unique ID of the user whose is_active status should be updated\n
    `active_request` (ActiveUserRequest):
    - `is_active` (bool): The new active status to assign to the user (true or false)\n
    `db` (Session, dependency): Injected SQLAlchemy database session\n
    `_` (User, dependency): Injected current admin user (unused, but enforces RBAC)

    **Returns:**\n
    `UserAdminResponse`(response_model):
    - `id` (int): The user ID
    - `email` (str): The user's email address
    - `is_active` (bool): Whether the user account is active
    - `role` (str): The user's role

    `Requires:` Admin privileges
    '''
    # Check if is active is part of payload. if not -> exception
    is_active = active_request.is_active
    if is_active is None:
        raise HTTPException(status_code=400, detail="Missing field: is_active")
    
    # Check if is_active is bool
    if not isinstance(is_active, bool):
        raise HTTPException(status_code=400, detail="is_active must be boolean.")
    
    # Check if user is registered in DB. If not -> Exception
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    
    # Update user active status
    user.is_active = is_active
    db.commit()
    db.refresh(user)
    
    return UserResponse(
        id=user.id,
        email=user.email,
        is_active=user.is_active,
        role=user.role,
    )