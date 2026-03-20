from src.db.models.ratings import Rating

RATE_MOV_1 = Rating(
    userId=1,
    movieId=1,
    rating=4.5,
    timestamp=1620000000,
)

RATE_MOV_2 = Rating(
    userId=2,
    movieId=1,
    rating=3.0,
    timestamp=1620000001,
)

RATE_MOV_3 = Rating(
    userId=3,
    movieId=1,
    rating=5.0,
    timestamp=1620000002,
)
