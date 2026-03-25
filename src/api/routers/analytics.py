from fastapi import APIRouter, Depends, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

from src.db.database_session import get_db

# Define router 
router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"],
)

@router.get("/db-information")
def get_db_information(db: Session = Depends(get_db)):
    """
    Endpoint to retrieve aggregated database statistics.

    This function executes SQL aggregation queries directly in the database
    to avoid loading large tables into memory.

    Parameters
    ----------
        db (Session): SQLAlchemy session provided by dependency injection

    Returns
    -------
        dict: Dictionary containing computed statistics
    """

    # SQL query to compute main statistics
    query = text("""
        SELECT 
            COUNT(DISTINCT "movieId") AS num_movies,
            COUNT(*) AS num_ratings,
            COUNT(DISTINCT "userId") AS num_users,
            AVG(rating) AS avg_rating
        FROM ratings;
    """)

    # Execute query and fetch single row result
    result = db.execute(query).mappings().first()

    # Additional queries for averages (more efficient than doing in Python)
    avg_user_query = text("""
        SELECT AVG(user_count) AS avg_ratings_per_user
        FROM (
            SELECT COUNT(*) AS user_count
            FROM ratings
            GROUP BY "userId"
        ) sub;
    """)

    avg_movie_query = text("""
        SELECT AVG(movie_count) AS avg_ratings_per_movie
        FROM (
            SELECT COUNT(*) AS movie_count
            FROM ratings
            GROUP BY "movieId"
        ) sub;
    """)

    avg_user = db.execute(avg_user_query).scalar()
    avg_movie = db.execute(avg_movie_query).scalar()

    # Compute sparsity
    total_possible = result["num_users"] * result["num_movies"]
    sparsity = 1 - (result["num_ratings"] / total_possible) if total_possible else 0

    return {
        "num_movies": result["num_movies"],
        "num_ratings": result["num_ratings"],
        "num_users": result["num_users"],
        "avg_rating": float(result["avg_rating"]),
        "avg_ratings_per_user": float(avg_user),
        "avg_ratings_per_movie": float(avg_movie),
        "sparsity": float(sparsity)
    }


@router.get("/rating-distribution")
def get_rating_distribution(db: Session = Depends(get_db)):
    """
    Endpoint to retrieve the distribution of ratings.

    This function computes how often each rating value occurs directly in the database
    using GROUP BY, which is much mor efficient than loading all ratings into memory.


    Parameters
    ----------
        db (Session): SQLAlchemy session provided by dependency injection

    Returns
    -------
        list[dict]: List of rating values with their corresponding counts
                    Example:
                    [
                        {"rating": 0.5, "count": 1200},
                        {"rating": 1.0, "count": 3400},
                        ...
                    ]
    """
    # SQL query to compute histogram of ratings
    query = text("""
        SELECT 
            rating,
            COUNT(*) AS count
        FROM ratings
        GROUP BY rating
        ORDER BY rating;
    """)

    # Execute query and fetch all results as dictionaries
    result = db.execute(query).mappings().all()

    # Convert Decimal / numeric types to Python float for JSON compatibility
    distribution = [
        {
            "rating": float(row["rating"]),
            "count": int(row["count"])
        }
        for row in result
    ]

    return distribution


@router.get("/ratings-head")
def get_ratings_head(
    n: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """
    Returns the first n rows of the ratings table.

    Parameters
    ----------
        n (int): Number of rows to return (default=10, max=100)
        db (Session): Database session

    Returns
    -------
        list[dict]: List of rating records
    """

    query = text(f"""
        SELECT "userId", "movieId", rating, timestamp
        FROM ratings
        LIMIT :n
    """)

    result = db.execute(query, {"n": n}).mappings().all()

    return result



@router.get("/movies-head")
def get_movies_head(
    n: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Returns the first n rows of the movies table.

    Parameters
    ----------
        n (int): Number of rows to return (default=10, max=100)
        db (Session): Database session

    Returns
    -------
        list[dict]: List of movie records
    """

    query = text(f"""
        SELECT "movieId", title, genres
        FROM movies
        LIMIT :n
    """)

    result = db.execute(query, {"n": n}).mappings().all()

    return result