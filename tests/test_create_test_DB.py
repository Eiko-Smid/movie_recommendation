import pytest
from fastapi.testclient import TestClient

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.main import app
from src.db.database_session import Base
from src.db.models.users import User
from src.db.models.ratings import Rating
from src.db.models.movies import Movie
from src.db.database_session import get_db

from src.api.security import hash_password
from src.api.role import UserRole

from tests.utils.db_test_user import (
    ADMIN_USER,
    DEV_USER,
    USER_USER
)

from tests.utils.db_test_ratings import (
    RATE_MOV_1,
    RATE_MOV_2,
    RATE_MOV_3
)

from tests.utils.db_test_movies import (
    MOVIE_1,
    MOVIE_2,
    MOVIE_3
)

TEST_DATABASE_URL = "sqlite:///./test.db"

# Create db connection
engine = create_engine(TEST_DATABASE_URL)
TestingSessionLocal = sessionmaker(bind=engine)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    '''
    Create test db for testing api endpoints.
    '''
    # Create schema
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()

    # Add test userdata to DB
    db.add_all([ADMIN_USER, DEV_USER, USER_USER])

    # Add test ratings to DB
    db.add_all([RATE_MOV_1, RATE_MOV_2, RATE_MOV_3])

    # Add test movies to DB
    db.add_all([MOVIE_1, MOVIE_2, MOVIE_3])

    db.commit()
    db.close()

    yield

    # CLean everything 
    Base.metadata.drop_all(bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


