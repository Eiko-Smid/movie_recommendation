import os 
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from dotenv import load_dotenv 

from fastapi import HTTPException, status


def get_db_url() -> str:
    load_dotenv()
    DB_URL = os.getenv("DB_URL")
    if not DB_URL:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection URL not found in environment variables."
        )
    return DB_URL


# Create global DB engine 
engine = create_engine(get_db_url(), pool_pre_ping=True)

# Create Session factory
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


# Base class for models
class Base(DeclarativeBase):
    pass


def get_db():
    # Create  DB session
    db = SessionLocal()
    try:
        # Return session
        yield db
    finally:
        # Close session on second call
        db.close()

