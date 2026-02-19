from fastapi import APIRouter, HTTPException, status, Request, Depends
from pydantic import BaseModel, Field
from typing import Optional, Sequence, List, Dict, Tuple

import logging

import pandas as pd

from src.models.als_movie_rec import (
    grid_search_advanced,
)
from src.db.database_session import engine
from src.db.db_requests import (
    refresh_mv,
    _load_full_histories_for_n_users,
    _load_full_mv_users,
    MV_NAME
) 

from src.api.schemas import (
    TrainRequest,
    TrainResponse,
    BestParameters,
)

from src.models.management import (
    prepare_training,
    did_model_improve,
    mlflow_log_run,
    update_champ_model
)

from src.db.models.users import User
from src.api.security import check_user_authorization
from src.api.role import UserRole


# Init logger
logger = logging.getLogger(__name__)

# _________________________________________________________________________________________________________
# Data loading functionality 
# _________________________________________________________________________________________________________

# def _load_data(train_param: TrainRequest) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     ''' 
#     Loads the ratings and movies CSV files into Pandas DataFrames.

#     This function is called at the beginning of the /train endpoint to load
#     the MovieLens ratings and movie metadata. It verifies that the files exist
#     and handles limited row loading based on the value of train_param.n_rows.

#     Parameters
#     ----------
#     train_param: TrainRequest
    # Training configuration containing 'n_users'.

#     Returns
#     -------
#     df_ratings: pd.DataFrame
#         Data frame were each rows contains a pair of (userId, movieId, rating, timestamp)
#     df_movies: pd.DataFrame
#         Data frame were each rows contains a pair of (movieId, title, genres)
#     '''
#     # Define file paths
#     data_path_ratings = "data/ml-20m/ratings.csv"
#     data_path_movies = "data/ml-20m/movies.csv"

#     # Load data
#     n_rows = train_param.n_rows
#     # Check for existing paths
#     if not os.path.exists(data_path_ratings):
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail=f"Ratings csv file not found. Path is:\n{data_path_ratings}"
#         )

#     if not os.path.exists(data_path_movies):
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail=f"Movies csv file not found. Path is:\n{data_path_movies}"
#         )
    
#     # Try to load data
#     try:
#         df_ratings = pd.read_csv(data_path_ratings, nrows= n_rows if n_rows > 0 else None)
#         df_movies = pd.read_csv(data_path_movies)
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=f"Failed to read CSVs: {e}"
#             )
    
#     return df_ratings, df_movies

    
def _load_data(train_param: TrainRequest) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load ratings + movies from DB (same logic as before in main).

    Uses the materialized view and respects `n_users` semantics.
    """
    n_users = int(train_param.n_users)

    try:
        # Refresh or create MV 
        refresh_mv()

        # Set n_users to default value if zero
        if n_users == 0:
            n_users = 500

        with engine.connect() as conn:            
            if n_users < 0:
                # Load all the data of MV
                logger.info("Training with full MV ('%s') ratings.", MV_NAME)
                df_ratings = _load_full_mv_users()
            else:
                # Load n_users data                
                df_ratings = _load_full_histories_for_n_users(n_users_target=n_users)
                logger.info("Training with partial MV ('%s') ratings", MV_NAME)

            # Load movies
            df_movies = pd.read_sql_query('SELECT "movieId", title, genres FROM movies', conn)
            logger.info("Loaded movies successfully.")

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to load data from database: {e}")

    return df_ratings, df_movies


# _________________________________________________________________________________________________________
# API Endpoints
# _________________________________________________________________________________________________________

router = APIRouter(prefix="/train", tags=["train"])

@router.post(
        "/refresh-mv"
)
def refresh_mv_endpoint(
    _: User = Depends(check_user_authorization(UserRole.ADMIN, UserRole.DEVELOPER)),
):
    '''
    Refreshes the Materialized View (all users > 5 ratings). This is needed such that the
    api train endpoint has access to the newest data which lives inside the materialized view.
    The endpoint should get called to frequently, once a day before training is enough.
    '''
    refreshed_concurrently = refresh_mv()
    return {"status": "ok", "concurrent": refreshed_concurrently}



@router.post(
        "/train_model",
        response_model=TrainResponse,
)
def train_endpoint(
    request: Request,
    train_param: TrainRequest,
    _: User = Depends(check_user_authorization(UserRole.ADMIN, UserRole.DEVELOPER)),
):
    '''
    Trains or updates the ALS recommendation model using the provided training parameters.

    The endpoint:
      1. Loads and prepares the movie rating data.
      2. Performs a three-stage grid search (`grid_search_advanced`) to find the best
         challenger model
      3. Compares the new model against the current Champion model (if it exists).
      4. Logs all parameters, metrics, and artifacts to MLflow.
      5. Updates the current champion model. After every update the production model
         gets updated automatically as well.

    Parameters
    ----------
    train_param : TrainRequest
        Request body containing the ALS and preprocessing parameters such as BM25 settings,
        ALS hyperparameters, and dataset options.

    Returns
    -------
    TrainResponse
        Object containing the best hyperparameters 'best_param' and corresponding evaluation
        metrics 'best_metrics' from the training run.
    '''
    # Load data
    df_ratings, df_movies = _load_data(train_param=train_param)

    # Prepare training 
    (
    df_ratings,
    train_csr,
    test_csr,
    test_csr_masked,
    mappings,
    movie_id_dict,
    popular_item_ids,
    ) = prepare_training(
        df_ratings,
        df_movies,
        train_param,
    )

    # Train model
    # Grid search
    model, metrics_ls, parameter_ls, best_idx, used_params = grid_search_advanced(
        train_csr=train_csr,
        test_csr=test_csr_masked,
        bm25_K1_list=train_param.als_parameter.bm25_K1_list,
        bm25_B_list=train_param.als_parameter.bm25_B_list,
        factors_list=train_param.als_parameter.factors_list,
        reg_list=train_param.als_parameter.reg_list,
        iters_list=train_param.als_parameter.iters_list,
        n_samples=12
    )

    # Extract best parameters & metrics
    best_param = BestParameters(
        best_K1=parameter_ls[best_idx]["bm25_K1"],
        best_B=parameter_ls[best_idx]["bm25_B"],
        best_factor=parameter_ls[best_idx]["factors"],
        best_reg=parameter_ls[best_idx]["reg"],
        best_iters=parameter_ls[best_idx]["iters"],
    )
    best_metrics = metrics_ls[best_idx]
    
    # Check if model has improved compared to current champ model
    improved, champ_model, champ_metrics, champ_params = did_model_improve(
        train_csr=train_csr,
        test_csr=test_csr_masked,
        best_metrics=best_metrics
    )

    # Log param. metrics, models
    if not improved and champ_model is not None:
        new_version, best_weighted_csr = mlflow_log_run(
            train_param=train_param,
            model=champ_model,
            used_grid_param=used_params,
            mappings=mappings,
            best_param=champ_params,
            best_metrics=champ_metrics,
            train_csr=train_csr,
            popular_item_ids=popular_item_ids,
            movie_id_dict=movie_id_dict
        )
    else:
       new_version, best_weighted_csr = mlflow_log_run(
            train_param=train_param,
            model=model,
            used_grid_param=used_params,
            mappings=mappings,
            best_param=best_param,
            best_metrics=best_metrics,
            train_csr=train_csr,
            popular_item_ids=popular_item_ids,
            movie_id_dict=movie_id_dict
        )         
    
    # Update the champ model
    update_champ_model(
        app=request.app,
        new_version=new_version,
        best_weighted_csr=best_weighted_csr
    )

    # Return train response
    return TrainResponse(best_param=best_param, best_metrics=metrics_ls[best_idx])
