from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status, Depends

from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError

from src.db.models.users import User
from src.db.database_session import get_db
from src.db.models.app_ratings import AppRating
from src.db.models.movies import Movie

from src.api.schemas import RateMovieRequest, RateMovieResponse
from src.api.security import check_user_authorization
from src.api.role import UserRole


router = APIRouter(prefix="/update_DB", tags=["rate_movie"])
@router.post(
    "/rate_movie",
    response_model=RateMovieResponse,
)
def rate_movie(
    request: RateMovieRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_user_authorization(UserRole.ADMIN, UserRole.DEVELOPER, UserRole.USER)),
):
    # Check if movie id exists
    movie_id_exists = db.query(Movie).filter(Movie.movieId == request.movie_id).first()
    if not movie_id_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"The given movie ID {request.movie_id} does not exists in DB. Please add movie first.",
        )
    
    # Create movie rating row and create if not existing, else update row
    timestamp = int(datetime.now(timezone.utc).timestamp())
    stmt = insert(AppRating).values(
        userId=current_user.id,
        movieId=request.movie_id,
        rating=request.rating,
        timestamp=timestamp
    ).on_conflict_do_update(
        index_elements=["userId", "movieId"],
        set_={"rating": request.rating, "timestamp": timestamp},
    )

    try:
        # Send command to sql
        db.execute(stmt)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Failed to save rating due to a database constraint violation. "
                "The referenced movie may not exist or the data is invalid. "
                "Please verify the movie_id and rating value."
            ),
        )
    
    return RateMovieResponse(
        message="Rating saved.",
        movie_id=request.movie_id,
        user_id=current_user.id,
        rating=request.rating,
        timestamp=timestamp
    )

