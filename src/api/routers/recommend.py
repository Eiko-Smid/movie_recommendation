from fastapi import Request, FastAPI, HTTPException, status, Depends, APIRouter
from typing import List, Optional, Sequence, Dict, Tuple

import pandas as pd

from src.api.schemas import (
    RecommendRequest,
    RecommendResponse
)

from src.models.management import (
    get_champion_model
)

from src.db.users import User
from src.api.security import check_user_authorization
from src.api.role import UserRole


router = APIRouter(prefix="/recommend", tags=["recommend"])
@router.post(
    "/recommend",
    response_model=RecommendResponse,
)
def recommend_endpoint(
    recom_param: RecommendRequest,
    champion_model = Depends(get_champion_model),
    _: User = Depends(check_user_authorization(UserRole.ADMIN, UserRole.DEVELOPER, UserRole.USER)),
):
    '''
    Generates a personalized movie recommendation if the given user is part of the matrix 
    the current champ model was trained with. If thats not the case but some initial 
    information about the users favorit movies are provided the recommendations are 
    getting computed on the fly. 
    If none of both is the case, the user get recommendations based of a list of popular 
    movies.

    Parameters
    ----------
    recom_param : RecommendRequest
        Class including the user id, the number movies to recommend and a list of movies
        the user likes. The list is for the case that the user id is not known.
    Returns
    -------
    RecommendResponse
        Endpoint returns the user_id, the ids of the recommended movies as well as the
        movie names.
    '''
    # Check if Champ model exists, if not raise Exception
    if champion_model is None:
        raise HTTPException(status_code=503, detail="Champion model not loaded yet")
    
    # Put request into df (Specified by MLFLow that model_input needs to be df or numpy.ndarray)
    df = pd.DataFrame([{
        "user_id": recom_param.user_id,
        "n_movies_to_rec": recom_param.n_movies_to_rec,
        "new_user_interactions": recom_param.new_user_interactions or []
    }])

    # Predict -> Movie recommendations
    df_movie_rec = champion_model.predict(df) 

    # Extract df data
    movie_ids = df_movie_rec.iloc[0]["movie_ids"]
    movie_titles = df_movie_rec.iloc[0].get("movie_titles", None) or []
    movie_genres = df_movie_rec.iloc[0].get("movie_genres", None) or []

    return RecommendResponse(
        user_id=recom_param.user_id,
        movie_ids=movie_ids,
        movie_titles=movie_titles,
        movie_genres=movie_genres
    )