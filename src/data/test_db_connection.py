import os
import pytest
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# Create engine
DB_URL = os.getenv('DB_URL')

@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skip DB test on CI")
def test_db_connection():
    assert DB_URL is not None, "DB_URL is not set"
    engine = create_engine(DB_URL)
    with engine.connect() as connection:
        assert connection is not None