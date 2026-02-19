from sqlalchemy import (
    Integer,
    Numeric,
    BigInteger,
    ForeignKey
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship
)
from typing import List

from src.db.database_session import Base


class Rating(Base):
    """
    ORM model for 'ratings' table.

    Columns
    -------
    userId : int
        ID of the user who rated the movie.
    movieId : int
        Foreign key referencing movies.movieId.
    rating : float
        Rating value (e.g., 3.5).
    timestamp : int
        Unix timestamp of rating event.
    """

    __tablename__ = "ratings"

    userId: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        nullable=False
    )

    movieId: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("movies.movieId", ondelete="CASCADE"),
        primary_key=True,
        nullable=False
    )

    rating: Mapped[float] = mapped_column(
        Numeric(2, 1),
        nullable=False
    )

    timestamp: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False
    )
