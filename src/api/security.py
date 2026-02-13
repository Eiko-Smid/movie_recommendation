from fastapi import Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordBearer

from sqlalchemy.orm import Session
from src.db.database_session import get_db

import os
import logging
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from passlib.context import CryptContext

from src.api.role import UserRole
from src.db.users import User
from src.db.database_session import SessionLocal


# Create configured hashing machine -> Hash pwd with this machine
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Load env vars for JWT
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRE_MIN", "30"))

oauth_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def hash_password(pwd: str) -> str:
    '''
    Hashes the given password with the defined pwd_context manager (hashing machine).
    
    Parameters
    ----------
    pwd: str
        The password that will be hashed
    
    Returns
    -------
    The hashed password.
    '''
    return pwd_context.hash(pwd)


def verify_password(pwd: str, hashed_pwd: str) -> bool:
    '''
    Verifys if the given plain password corresponds to the given hashed pwd.
    
    Parameters
    ----------
    plain_pwd: str
        Password in raw text.
    hashed_pwd: str
        Hashed password.
        
    Returns
    -------
    True if plain_pwd and hashed password belong together, otherwise false.
    '''
    return pwd_context.verify(pwd, hashed_pwd)


def create_access_token(subject: str, role: UserRole) -> str:
    '''
    Creates an JWT access token for the given subject.

    Parameters
    ----------
    subject: str
        Subject to give the token to, here users email.

    Returns
    -------
    access_token: str
        The jwt access token for the subject.
    '''
    # Define expiration date
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MIN) 

    # Define payload
    payload = {
        "sub": subject,         # Email
        "role": role.value,
        "exp": expire,
    }

    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """
    Decodes a jwt token and returns the username (email) it belongs to.

    Parameters
    ----------
    token : str
        The jwt access token for a specific subject.

    Returns
    -------
    payload: str
        The payload of the decoded JWT token.
    """
    # Decode the jwt token into a dict with keys sub, role and exp.
    payload = jwt.decode(
        token=token,
        key=JWT_SECRET,
        algorithms=[JWT_ALGORITHM],
    )

    # Extracts the sub value from the dict -> username = email
    sub = payload.get("sub")

    # Checks if username could be extracted
    if not sub:
        raise JWTError("Missing subject (sub) claim.")
    
    return payload


def get_current_user(
        payload: str = Security(oauth_scheme),
        db: Session = Depends(get_db),
) -> User:
    '''
    Gets a jwt access token and a db session object. First it decodes the given access token. 
    If the extracted email is part of the DB, then the function returns the user information.
    Otherwise raises an HTTP Exception.
    
    Parameters
    ----------
    token: str
        JWT access token.
    db: Session
        A DB session object to access the DB trough SQLalchemy.

    Returns
    ----------
    user : User
        A User object containing the user information that correspondes to the 
        given toke, if found. Else HTTPException.        
    '''
    # Checks if token is valid else, Exception
    try:
        payload = decode_token(token=payload)
        email = payload.get("sub")
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Search if email exists in DB
    user = db.query(User).filter(User.email == email).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive.")
    
    return user


def check_user_authorization(*allowed_roles: UserRole):
    '''
    Checks if the current user is a valid user and if his role matches the given 
    list of allowed roles. Only the allowed roles are valid.
    
    This function acts as a dependency factory. It returns a callable that:
        - Retrieves the current authenticated user
        - Verifies that the user's role is included in the given allowed roles
        - Raises HTTP 403 if the role is not permitted

    Parameters
    ----------
    *allowed_roles : UserRole
        One or more roles that are allowed to access the protected endpoint.
        Multiple roles can be passed as positional arguments, e.g.:
            check_user_rights(UserRole.ADMIN)
            check_user_rights(UserRole.DEVELOPER, UserRole.ADMIN)
    
    Returns
    ----------        
    callable
        A FastAPI dependency function that:
            - Injects the authenticated User
            - Validates the user's role
            - Returns the User object if authorized
            - Raises HTTPException(403) if unauthorized
    '''
    def _checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permission."
            )
        return current_user
    return _checker


def init_authorization():
    '''
    Initializes the application's administrator account in the database.

    This function is idempotent and safe to run at application startup. It opens a
    dedicated database session and ensures there is an active user with the
    UserRole.ADMIN role by performing the following steps:

      1. If any active user with role UserRole.ADMIN exists, initialization is skipped.
      2. Otherwise, it reads ADMIN_EMAIL and ADMIN_PASSWORD from environment
         variables. If either variable is missing, initialization is skipped and a
         warning is logged.
      3. If a user with ADMIN_EMAIL already exists, that user's role is updated to
         UserRole.ADMIN and the change is committed.
      4. If no user exists with ADMIN_EMAIL, a new active User is created with the
         provided email, the password is hashed via hash_password(), assigned
         role UserRole.ADMIN, and persisted to the database.
    '''
    try:
        # Start DB session
        db = SessionLocal()

        # Check if an admin user already exists in the DB
        admin_exists = db.query(User).filter(User.role == UserRole.ADMIN).first()
        
        # Skip initialization if admin already exists
        if admin_exists:
            logging.info("Admin user already exists. Skipping admin initialization.")
            return

        # Load admin credentials from env file
        admin_email = os.getenv("ADMIN_EMAIL")
        admin_pwd = os.getenv("ADMIN_PASSWORD")

        if not admin_email or not admin_pwd:
            logging.warning("No admin email or password found in env file. Admin initialization skipped.")
            return
        
        # Check if admin user already exists in DB
        user = db.query(User).filter(User.email == admin_email).first()

        # If user exists -> give user admin role
        if user:
            user.role = UserRole.ADMIN
            db.commit()
            db.refresh(user)
        else:
            # Create new user with admin credentials 
            new_admin = User(
                email=admin_email,
                hashed_password=hash_password(admin_pwd),
                is_active=True,
                role=UserRole.ADMIN,
            )
            db.add(new_admin)
            db.commit()       
    finally:
        db.close()
