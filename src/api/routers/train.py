from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, Sequence, List, Dict, Tuple
import pandas as pd

from src.models.als_movie_rec import (
    prepare_data,
    build_movie_id_dict,
    get_popular_items,
    grid_search_advanced,
    ALS_Metrics,
)
from src.db.database_session import engine
from src.db.db_requests import (
    _create_mv_if_missing,
    _load_full_histories_for_n_users,
    refresh_mv,
) 

from src.api.schemas import (
    TrainRequest,
    TrainResponse,
    BestParameters,
)



def _load_data(train_param: TrainRequest) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load ratings + movies from DB (same logic as before in main).

    Uses the materialized view and respects `n_users` semantics.
    """
    n_users = int(train_param.n_users)

    try:
        _create_mv_if_missing()

        if n_users == 0:
            n_users = 500

        with engine.connect() as conn:
            if n_users < 0:
                df_ratings = pd.read_sql_query('SELECT "userId", "movieId", rating, "timestamp" FROM ratings', conn)
            else:
                df_ratings = _load_full_histories_for_n_users(n_users_target=n_users)

            df_movies = pd.read_sql_query('SELECT "movieId", title, genres FROM movies', conn)

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to load data from database: {e}")

    return df_ratings, df_movies


def prepare_training(df_ratings: pd.DataFrame, df_movies: pd.DataFrame, train_param: TrainRequest):
    """Prepare train/test CSRs, mappings and popular items for training."""
    train_csr, test_csr, test_csr_masked, mappings, evaluation_set = prepare_data(
        df=df_ratings, pos_threshold=train_param.pos_threshold
    )

    movie_id_dict = build_movie_id_dict(df_movies)

    popular_item_ids = get_popular_items(
        df=df_ratings, top_n=train_param.n_popular_movies, threshold=train_param.pos_threshold
    )

    return (
        df_ratings,
        train_csr,
        test_csr,
        test_csr_masked,
        mappings,
        movie_id_dict,
        popular_item_ids,
    )


# _________________________________________________________________________________________________________
# API Endpoints
# _________________________________________________________________________________________________________

router = APIRouter(prefix="/train", tags=["train"])

@router.post(
        "/refresh-mv",
        summary="Refreshes the Materialized View (all users > 5 ratings)",
        description=(
        "When called the materialized view inside the data base gets refreshed."
        "This is needed such that the api train endpoint has access to the newest"
        "data which lives inside the materialized view."
        "The endpoint should get called to frequently, once a day before training"
        "is enough."
    ),
    response_description="Status (ok if it worked). and the way the mv was refreshed." \
    "Either concurrently (new mv gets created while old one still useable -> parallel) "
    "or not concurrently (new mv gets created while old one still not useable)."
)
def refresh_mv_endpoint():
    refreshed_concurrently = refresh_mv()
    return {"status": "ok", "concurrent": refreshed_concurrently}



@router.post(
        "/train_model",
        response_model=TrainResponse,
        summary="Train or update the ALS model",
)
def train_endpoint(train_param: TrainRequest):
    '''
    Trains or updates the ALS recommendation model using the provided training parameters.

    The endpoint:
      1. Loads and prepares the movie rating data.
      2. Performs a three-stage grid search (`grid_search_advanced`) to find the best
         challanegr model
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
    # Load & prepare
    df_ratings, df_movies = _load_data(train_param=train_param)

    (
        df_ratings,
        train_csr,
        test_csr,
        test_csr_masked,
        mappings,
        movie_id_dict,
        popular_item_ids,
    ) = prepare_training(df_ratings, df_movies, train_param)

    # Grid search (delegates to model logic)
    model, metrics_ls, parameter_ls, best_idx, used_params = grid_search_advanced(
        train_csr=train_csr,
        test_csr=test_csr_masked,
        bm25_K1_list=train_param.als_parameter.bm25_K1_list,
        bm25_B_list=train_param.als_parameter.bm25_B_list,
        factors_list=train_param.als_parameter.factors_list,
        reg_list=train_param.als_parameter.reg_list,
        iters_list=train_param.als_parameter.iters_list,
        n_samples=12,
    )

    best_param = BestParameters(
        best_K1=parameter_ls[best_idx]["bm25_K1"],
        best_B=parameter_ls[best_idx]["bm25_B"],
        best_factor=parameter_ls[best_idx]["factors"],
        best_reg=parameter_ls[best_idx]["reg"],
        best_iters=parameter_ls[best_idx]["iters"],
    )
    best_metrics = metrics_ls[best_idx]

    # Compare to champion and log
    improved, champ_model, champ_metrics, champ_params = did_model_improve(
        train_csr=train_csr, test_csr=test_csr_masked, best_metrics=best_metrics
    )

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
            movie_id_dict=movie_id_dict,
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
            movie_id_dict=movie_id_dict,
        )

    # Promote & update in-memory Champion
    update_champ_model(new_version=new_version, best_weighted_csr=best_weighted_csr)

    return TrainResponse(best_param=best_param, best_metrics=best_metrics)

