from src.db.models.users import User
from src.api.role import UserRole
from src.api.security import hash_password


ADMIN_USER = User(
    id=1,
    email="admin@test.com",
    hashed_password=hash_password("admin"),
    is_active=True,
    role=UserRole.ADMIN,
)


DEV_USER = User(
    id=2, 
    email="dev@test.com",
    hashed_password=hash_password("dev"),
    is_active=True,
    role=UserRole.DEVELOPER,
)


USER_USER = User(
    id=3,
    email="user@test.com",
    hashed_password=hash_password("user"),
    is_active=True,
    role=UserRole.USER,
)