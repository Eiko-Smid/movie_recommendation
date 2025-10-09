import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# Create engine
DB_URL = os.getenv('DB_URL')
engine = create_engine(DB_URL)

try:
    with engine.connect() as connection:
        print("Connection successful!")
except Exception as e:
    print("Connection failed:", e)