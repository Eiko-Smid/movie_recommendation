from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from sqlalchemy.orm import Session

from src.db.database_session import get_db
from src.db.users import User
from src.api.schemas import UserCreate, Token
from src.api.security import hash_password, verify_password, create_access_token


# Init route obj
router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    """
    Get's the user email and password (UserCreate) and creates a new user inside
    the DB, if the email doesn't already exist.

    **Parameters**:\n
    `payload` (UserCreate): The user email and password.\n

    **Returns**:\n
    `message` (str): Success message indicating user creation.
    """
    # Check if email already exists in DB
    email_exists = db.query(User).filter(User.email == payload.email).first()

    # Skip creation if email exists
    if email_exists:    
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already exists.")
    
    # Create new user obj
    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        is_active=True,
    )

    # Add user to DB
    db.add(user)
    db.commit()

    return {"message": "Created user successfully."}


@router.post("/token", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """
    Login endpoint (OAuth2 password flow). Returns a JWT access token if user email and 
    password are valid.

    **Returns**:\n
    `Token`(response_model): 
    - `access_token` (str): JWT access token for authentication.\n
    - `token_type` (str): Type of the token, typically "bearer".            
    """
    # Check if username (email) is part of DB table
    user = db.query(User).filter(User.email == form_data.username).first()

    # Raise error if user password is invalid 
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bad credentials."
        )
    
    # Create an jwt access token after validating credentials.
    token = create_access_token(subject=user.email, role=user.role)

    return Token(access_token=token)

