import os

from dotenv import load_dotenv
from fastapi import HTTPException, status
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


def get_db_url() -> str:
    """
    Returns the database URL depending on the environment.

    TESTING:
        Uses SQLite (file-based, no external DB required)

    PRODUCTION:
        Uses DB_URL from environment variables
    """
    # Get testing env flag 
    load_dotenv()
    TESTING = os.getenv("TESTING", "false").lower() == "true"

    # Create DB URL for sql light in the testing environment
    if TESTING:
        # SQLite fallback for CI / local container testing
        return "sqlite:///./test.db"

    # Get postgre DB url from env var
    DB_URL = os.getenv("DB_URL")
    if not DB_URL:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection URL not found in environment variables.",
        )

    return DB_URL


def init_db():
    """
    Initializes the database engine and session factory.

    MUST be called at runtime (e.g., inside FastAPI lifespan).
    Avoids connection attempts during module import.
    """

    # Get DB URL and create engine
    db_url = get_db_url()
    engine = create_engine(db_url, pool_pre_ping=True)

    # Create session factory to enable DB sessions
    SessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    return engine, SessionLocal


# Base class for SQLAlchemy models
class Base(DeclarativeBase):
    pass


def get_db():
    """
    FastAPI dependency that provides a database session.

    Yields:
        SQLAlchemy session

    Raises:
        RuntimeError if DB not initialized
    """
    if SESSION_LOCAL is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    db = SESSION_LOCAL()
    try:
        yield db
    finally:
        db.close()

        
# Init engine and Session local objects
ENGINE, SESSION_LOCAL = init_db()