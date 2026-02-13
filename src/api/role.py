from enum import Enum

class UserRole(str, Enum):
    USER = "user"
    DEVELOPER = "developer"
    ADMIN = "admin"

