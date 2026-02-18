from sqlalchemy import Integer, ForeignKey, Column, Numeric, BigInteger

from src.db.database_session import Base


class AppRating(Base):
    __tablename__ = "app_ratings"

    userId = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    movieId = Column(Integer, ForeignKey('movies."movieId"', ondelete="CASCADE"), primary_key=True)
    rating = Column(Numeric(2, 1), nullable=False)
    timestamp = Column(BigInteger, nullable=False)
