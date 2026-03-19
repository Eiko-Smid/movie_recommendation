import os
# Create DB URL for pytest -> Ensures sqlight DB is used instead of postgreSQL DB
TEST_DATABASE_URL = "sqlite:///./test.db"
os.environ["DB_URL"] = TEST_DATABASE_URL

import pytest
from fastapi.testclient import TestClient

from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import pandas as pd

from src.api.main import app
from src.db.database_session import Base
from src.db.models.users import User
from src.db.models.ratings import Rating
from src.db.models.movies import Movie
from src.db.database_session import get_db
from src.db.db_requests import get_user_id_offset

from src.api.security import hash_password
from src.api.role import UserRole

from src.models.management import get_champion_model

from tests.utils.db_test_user import (
    ADMIN_USER,
    DEV_USER,
    USER_USER,
    INACTIVE_ADMIN_USER
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
    db.add_all([ADMIN_USER, DEV_USER, USER_USER, INACTIVE_ADMIN_USER])

    # Add test ratings to DB
    db.add_all([RATE_MOV_1, RATE_MOV_2, RATE_MOV_3])

    # Add test movies to DB
    db.add_all([MOVIE_1, MOVIE_2, MOVIE_3])

    db.commit()
    db.close()

    yield

    # CLean everything 
    Base.metadata.drop_all(bind=engine)



class DummyModel():
    '''
    Class to simulate a trained model that returns movie recommendations. 
    '''
    def predict(self, model_input: pd.DataFrame):
        '''
        Get's the model input and returns a data frame consisting of dummy data. Imitates
        the real model.predict method. 

        Parameters
        ----------
        model_input : pd.DataFrame
            DataFrame with columns:
            - user_id: int
            - n_movies_to_rec: int (optional)
            - new_user_interactions: list[int] (optional)

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns:
            - movie_ids: list[int]
            - movie_titles: list[str]
        '''
        movie_ids = []
        movie_titles = []
        movie_genres = []
        rows = []

        for _, row in model_input.iterrows():
            # Extract information
            # user_id: int = int(row["user_id"])
            n_rec: int = int(row.get("n_movies_to_rec", 5))
            # new_inter: Optional[list[int]] = row.get("new_user_interactions", None)

            # Simulate movie data
            for i in range(n_rec):
                movie_ids.append(i)
                movie_titles.append("title")
                movie_genres.append("genres")

            # Stack data together
            rows.append(
                {
                    "movie_ids": movie_ids,
                    "movie_titles": movie_titles,
                    "movie_genres": movie_genres,
                }
            )
        
        return pd.DataFrame(rows)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


def override_get_champion_model():
    '''
    Simulates the get_champion_model function, but this one returns a dummy model 
    instead.
    '''
    return DummyModel()


def override_user_id_offset():
    user_id_offset = 0
    return user_id_offset


@pytest.fixture
def client():
    # Overwrite fuction dependencies with test versions
    app.dependency_overrides[get_db] = override_get_db
    # app.dependency_overrides[get_champion_model] = override_get_champion_model
    app.dependency_overrides[get_user_id_offset] = override_user_id_offset

    # Override app.state
    app.state.champion_model_version = "test_version"
    app.state.champion_model = DummyModel()

    return TestClient(app)


