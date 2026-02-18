from sqlalchemy import Column, Integer, String
from src.db.database_session import Base


class Movie(Base):
    __tablename__ = "movies"

    movieId = Column("movieId", Integer, primary_key=True)
    title = Column(String)
    genres = Column(String)
