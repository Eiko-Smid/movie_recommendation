from __future__ import annotations
import logging
import os, requests
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import tempfile

from sqlalchemy import text
# from sqlalchemy.engine import Engine

from dotenv import load_dotenv 

from fastapi import FastAPI, HTTPException, status, Query, Request, APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from fastapi.requests import Request
from contextlib import asynccontextmanager

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple, Mapping, Iterable, Any, TypedDict, Callable

from datetime import datetime

from math import ceil
import numpy as np
import pandas as pd

import json
import hashlib
import platform
import zlib

import mlflow
from mlflow import MlflowClient
# from mlflow.tracking import MlflowClient
import joblib
from mlflow.pyfunc import PythonModel

from scipy.sparse import coo_matrix, csr_matrix, save_npz, load_npz
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight

from time import time, sleep

import random

# Import ALS recommend functionality
from src.models.als_movie_rec import (
    Mappings,
    ALS_Metrics,
    prepare_data,
    build_movie_id_dict,
    get_popular_items,
    als_grid_search,
    recommend_item,
    get_movie_names,
    get_movie_metadata,
    evaluate_als,
    grid_search_advanced
)

# Import sql request code
from src.data.database_session import engine
from src.data.db_requests import (
    _create_mv_if_missing,
    _load_full_histories_for_n_users,
    refresh_mv
)


# _________________________________________________________________________________________________________
# Global settings
# _________________________________________________________________________________________________________
CHAMP_STORE_DIR = Path("champ_store")
CHAMP_STORE_DIR.mkdir(parents=True, exist_ok=True)
CHAMPION_TRAIN_CSR_PATH = CHAMP_STORE_DIR / "champion_train_csr.npz"

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
# EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "als_movie_recommendation")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "als_movie_rec")
MODEL_NAME = os.getenv("MODEL_NAME", "als_model_versioning")

USE_ALIASES = True  # True = prefer aliases (Champion). False = use stages (Production).

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)

CHAMP_MODEL: Optional[mlflow.pyfunc.PyFuncModel] = None

client = MlflowClient()



# _________________________________________________________________________________________________________
# Helper functions and classes 
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
    """
    Creates a materialized view of all users which rated 'MIN_N_USER_RATINGS'or more movies,
    if not already existing. 
    Then requests the DB to get n_rows users which fulfil the condition. This ensures that we
    only get meaningful users. 
    Also requests the movies table (id, name, genres) from the db. 
    Stores both requests in two separated df's and returns them.

    Parameters
    ----------
    train_param:
        Training configuration containing 'n_users'.
            n_users > 0: Load all ratings from the n_users randomly
            n_users = 0: Load ratings from 500 (default) users 
            n_users < 0: Load all ratings, no mv filtered.

    Returns
    -------
    df_ratings: pd.DataFrame
        Data frame were each rows contains a pair of (userId, movieId, rating, timestamp)
    df_movies: pd.DataFrame
        Data frame were each rows contains a pair of (movieId, title, genres)

    """
    n_users = int(train_param.n_users)

    try:
        # Creates a materialized view -> table of all users ids > 5 movie ratings
        _create_mv_if_missing()

        # Use default value for zero case
        if n_users == 0:
            n_users = 500

        with engine.connect() as conn:
            if n_users < 0:
                # Load full dataset without filtering                
                df_ratings = pd.read_sql_query(
                        'SELECT "userId", "movieId", rating, "timestamp" FROM ratings',
                        conn,
                    )
            else:
                # Load n_users 
                df_ratings = _load_full_histories_for_n_users(n_users_target=n_users)
            
            # Load movies            
            df_movies = pd.read_sql_query(
                'SELECT "movieId", title, genres FROM movies', conn
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to load data from database: {e}"
        )
    
    return df_ratings, df_movies



def csr_fingerprint(X) -> str:
    h = 0
    for arr in (X.indptr, X.indices, X.data):
        h = zlib.crc32(arr.view(np.uint8), h)
    return f"{h & 0xffffffff:08x}"


def prepare_training(
        df_ratings: pd.DataFrame,
        df_movies: pd.DataFrame,
        train_param: TrainRequest
    ) -> Tuple[
        pd.DataFrame,
        csr_matrix,
        csr_matrix,
        Mappings,
        Dict[int, str],
        List[int]
    ]:
    '''
    Prepares the data for ALS model training.

    Prepares the data (clean nans, train/test/split/test csr filtering) and builds supporting
    lookup structures (ID mappings, movie ID dictionary, and a list of popular movies for 
    cold-start recommendations).

    Parameters
    ----------
    df_ratings : pd.DataFrame
        The ratings dataset loaded
    df_movies : pd.DataFrame
        The movies dataset loaded via _load_data().
    train_param : TrainRequest
        Training configuration containing 'pos_threshold' and 'n_popular_movies'.

    Returns
    -------
    df_ratings: pd.DataFrame
        Data frame were each rows contains a pair of (userId, movieId, rating, timestamp)
    train_csr: csr_matrix
        The train csr matrix (user-item) used for training the model.
    test_csr: csr_matrix
        The original test csr matrix.
    test_csr_masked: csr_matrix
        The filtered csr matrix used for evaluating the model. All train rows that have
        less than 5 train entries and 1 test entries will be set to zero. The zero lines
        will be automatically ignored when computing the evaluation metrics.
    mappings: Mappings
        Stores the relevant mappings needed to transfer the df user/item ids to the 
        user/item ids of the csr matrices.
    movie_id_dict: Dict[int, str]
        Lookup table which is a set of tuples were each tuple is a pair of the orginal movie
        id and the movie id in csr format.
    popular_item_ids: List[int]
        List of popular movie that are used for the case that the user is not know in the 
        data and that there is no information of movies the user likes. Solves the cold
        start problem in the recommendation part.
    '''
    # Only use random subset of df
    # perc = 0.8
    # n_samples = int(df_ratings.shape[0] * perc)
    # df_ratings = df_ratings.sample(n=n_samples, random_state=np.random.randint(0, 1_000_000))

    # Set up model training
    # Prepare data
    train_csr, test_csr, test_csr_masked, mappings, evaluation_set = prepare_data(
        df=df_ratings,
        pos_threshold= train_param.pos_threshold,
    )
    
    # Print fingerprints of csr amtrices to see if they stay consitent through runs
    # print("train_csr_hash:", csr_fingerprint(train_csr))
    # print("test_csr_hash:",  csr_fingerprint(test_csr))

    # Compute item-movie dict for quick lookup
    movie_id_dict = build_movie_id_dict(df_movies)

    # Get popular items for the case that a new user occurs that hasn't watched any movies by now.
    popular_item_ids = get_popular_items(
        df=df_ratings,
        top_n=train_param.n_popular_movies,
        threshold=train_param.pos_threshold)
    print(f"\nShape of popular_item_ids is {len(popular_item_ids)}")

    return df_ratings, train_csr, test_csr, test_csr_masked, mappings, movie_id_dict, popular_item_ids


def mlflow_log_run(
        train_param: TrainRequest,
        model: AlternatingLeastSquares,
        used_grid_param: Sequence[dict[int, float]],
        mappings: Mappings,
        best_param: BestParameters, 
        best_metrics: ALS_Metrics,
        train_csr: csr_matrix,
        popular_item_ids: List[int],
        movie_id_dict: Dict[int, str],
        # improved: bool
    ) -> Tuple[
        str,
        csr_matrix
    ]:
    '''
    Logs parameters, metrics, and artifacts to MLflow and registers the model.

    Creates a new MLflow run, logs the full grid search configuration, the best 
    parameters, metrics, and a serialized model state. Then registers the
    model version, tags it with evaluation metrics, and retrieves the current 
    Champion’s parameters (if any) for comparison.

    Parameters
    ----------
    train_param : TrainRequest
        The full training configuration used for this run.
    model : AlternatingLeastSquares
        The trained ALS model.
    used_grid_param: Sequence[dict[int, float]]
        The grid parameters that have actually been used for training the model.
    mappings : Mappings
        Stores the relevant mappings needed to transfer the df user/item ids to the 
        user/item ids of the csr matrices.
    best_param : BestParameters
        The best parameter combination found during grid search.
    best_metrics : ALS_Metrics
        The metrics for the best-performing model (precision@K, MAP@K).
    train_csr : csr_matrix
        The unweighted training CSR matrix.
    popular_item_ids : List[int]
        List of popular movies for cold-start users.
    movie_id_dict : Dict[int, str]
        Mapping of movieId to movie title.

    Returns
    -------
    new_version: str
        The new MLflow model version.
    best_weighted_csr: csr_matrix
        The BM25-weighted train matrix.
    '''
    # Define run name beased on date-time
    run_name = f"train | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | n_rows={train_param.n_users}"
    # Start MLFlow run to log metrics and model
    with mlflow.start_run(run_name=run_name):
        # Log the whole grid as a single JSON param (correct API: log_param) -> better overview
        mlflow.log_param(
            "search_space_json",
            json.dumps({
                "bm25_K1_list": list(train_param.als_parameter.bm25_K1_list),
                "bm25_B_list": list(train_param.als_parameter.bm25_B_list),
                "factors_list": list(train_param.als_parameter.factors_list),
                "reg_list": list(train_param.als_parameter.reg_list),
                "iters_list": list(train_param.als_parameter.iters_list),
            })
        )
        # Log used grid search param
        mlflow.log_dict(
            dictionary= {"Used_grid_parameter": used_grid_param},
            artifact_file="used_grid_param.json"
        )

        # Log best params from grid search
        mlflow.log_dict(best_param.model_dump(), "best_params.json")

        # Log metrics of model training
        mlflow.log_metrics({
            "prec_at_k":float(best_metrics.prec_at_k),
            "map_at_k":float(best_metrics.map_at_k)
        })

        # Create path to store model artifacts. The pyfunc model will need this later on!
        artifacts_dir = Path("artifacts_tmp")
        artifacts_dir.mkdir(parents=True, exist_ok=True)     
        artifacts_path = artifacts_dir / "model_state.joblib"
        
        # Recompute the BM25-weighted matrix that matches the *best* params
        best_weighted_csr = bm25_weight(train_csr, K1=best_param.best_K1, B=best_param.best_B).tocsr()

        # Build a full state (so colleagues loading via MLflow get an immediately-usable model)
        # TODO: Delete the train_csr matrix from the model state (too large -> too slow!)
        state_obj = Model_State(
            model=model,
            mappings=mappings,
            popular_item_ids=popular_item_ids,
            movie_id_dict=movie_id_dict
        )
        joblib.dump(state_obj, artifacts_path)

        # Log pyfunc model that loads our state (NOTE: artifact key is "state_path")
        mlflow.pyfunc.log_model(
            # name="model",
            artifact_path="model",
            python_model=ALSRecommenderPyFunc(),
            artifacts={
                "state_path": str(artifacts_path)   # <<< critical: matches load_context
            },
            pip_requirements=[
                "implicit",
                "scikit-learn",
                "scipy",
                "pandas",
                "numpy",
                "cloudpickle",
                "joblib",
            ],
        )
        # get run id from current run
        run_id = mlflow.active_run().info.run_id

    # Register the model -> Model is visible under models 
    registered = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=MODEL_NAME
    )

    new_version = registered.version

    # Wait for model version to be ready
    for _ in range(40):
        mv = client.get_model_version(MODEL_NAME, new_version)
        if mv.status == "READY":
            break
        sleep(0.25)

    # Store model metrics as tags
    client.set_model_version_tag(MODEL_NAME, new_version, "prec_at_k", str(float(best_metrics.prec_at_k)))
    client.set_model_version_tag(MODEL_NAME, new_version, "map_at_k", str(float(best_metrics.map_at_k)))

    return new_version, best_weighted_csr


def did_model_improve(
        train_csr: csr_matrix,
        test_csr: csr_matrix,
        best_metrics: ALS_Metrics,
        improve_threshold: float = 0.002
    ) -> Tuple[
        bool,
        Optional[AlternatingLeastSquares],
        Optional[ALS_Metrics],
        Optional[BestParameters]
    ]:
    '''
    Compares the new model against the retrained Champion model. 

    If no champion model exists the improved parameter will automatically be set to true.
    Otherwise it retrains a new als instance with the parameters of the champion model 
    and evaluates the model. The resulting map_at_k values is then compared to the map_at_k
    value of the challenger model. 

    If the challenger models wins, the improved parameter is set to true. 

    Parameters
    ----------
    train_csr: csr_matrix
        User–item training matrix used to fit each ALS model.
    test_csr: csr_matrix
        User–item test matrix used to evaluate each model configuration.
    best_metrics : ALS_Metrics
        The metrics of the challanger model.
    improve_threshold: float
        Only if the metrics of the challenger model is bigger than the metric of the champ
        model + this threshold, the im,proved flag is set to true.

    Returns
    -------
    improved: bool
        True if the new model performs better (higher MAP@K) or if no Champion exists,
        False otherwise.
    champ_model: Optional[AlternatingLeastSquares]
        A new instance of ALS trained with the best parameters of the current Champ model.
    champ_metrics: Optional[ALS_Metrics]
        The metrics of the champ model (prec_@_k, map_@_k)
    champ_params: Optional[BestParameters]
        The parameter combination the champ model was trained with.
    '''
    champ_model = None
    champ_metrics = None
    champ_params = None

    # Load best params from champ version
    champ_params = _load_champion_params(model_name=MODEL_NAME)

    if champ_params is None:
        # Set model improved flag to true if not champ model currently exists 
        champ_map = None
        improved = True
    else:
        # Retrain champ model on new data using the old best params -> Only one training
        print("\nRetrain the ALS model with champ parameters:\n")
        for i in range(3):
            champ_model, champ_metrics_list, champ_params_list, champ_idx, actual_params = als_grid_search(
                train_csr=train_csr,
                test_csr=test_csr,
                bm25_K1_list=[champ_params["bm25_K1"]],
                bm25_B_list=[champ_params["bm25_B"]],
                factors_list=[champ_params["factors"]],
                reg_list=[champ_params["reg"]],
                iters_list=[champ_params["iters"]]
            )
            
        # champ_prec = champ_metrics[champ_idx].prec_at_k
        champ_metrics = champ_metrics_list[champ_idx]
        champ_map = champ_metrics_list[champ_idx].map_at_k
        # model_prec = best_metrics.prec_at_k
        model_map = best_metrics.map_at_k
        # Get champ params
        champ_params = BestParameters(
            best_K1=champ_params_list[champ_idx]["bm25_K1"],
            best_B=champ_params_list[champ_idx]["bm25_B"],
            best_factor=champ_params_list[champ_idx]["factors"],
            best_reg=champ_params_list[champ_idx]["reg"],
            best_iters=champ_params_list[champ_idx]["iters"],
        )

        # Decide new model is better than old champ model
        if model_map > (champ_map + improve_threshold):
            improved = True
        else: 
            improved = False
    
    return improved, champ_model, champ_metrics, champ_params


def update_champ_model(
        new_version: str,
        best_weighted_csr: csr_matrix
    ):
    """
    Promotes a given model version to 'Champion' and updates global state.

    Marks the MLflow model version with the 'Champion' alias, stores its
    training matrix to disk for future predictions, and loads the new
    Champion model into memory for serving via the /recommend endpoint.

    Parameters
    ----------
    new_version : str
        The MLflow model version number to promote.
    best_weighted_csr : csr_matrix
        The BM25-weighted training matrix of the new Champion model.

    Returns
    -------
    None
    """
    # Mark new model as Champ model
    client.set_registered_model_alias(MODEL_NAME, "Champion", new_version)

    # Save the csr matrix that has been used to train the new champ model -> Predictions enabled
    TRAIN_CSR_STORE.save(best_weighted_csr)

    # Load the new champ model
    global CHAMP_MODEL
    CHAMP_MODEL = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@Champion")


class ALSRecommenderPyFunc(PythonModel):
    """
    The trained model saved with joblib can be loaded with 'mlflow.pyfunc.load_model'. This 
    model can use this extended logic. Meaning the model itself after loading contains the 
    'predict' method from this class.

    Exp.:
        # Load model 
        model = mlflow.pyfunc.load_model("runs:/<run_id>/model")
        # Predict the 
        model.predict(df)
    """
    def load_context(self, context: mlflow.pyfunc.model.PythonModelContext) -> None:
            """
            Loads the serialized Model_State object (stored as a joblib file)
            from the MLflow artifacts when the model is loaded.

            Parameters
            ----------
            context : mlflow.pyfunc.model.PythonModelContext
                MLflow context object providing artifact paths.
            """
            state_path: str = context.artifacts["state_path"]
            self._state: Model_State = joblib.load(state_path)


    # model_input: DataFrame with columns: user_id (int), n_movies_to_rec (int, optional),
    # new_user_interactions (list[int], optional)
    def predict(
            self,
            context: mlflow.pyfunc.model.PythonModelContext,
            model_input: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Uses the loaded model state to generate movie recommendations
        for each user in the input DataFrame.

        Parameters
        ----------
        context : mlflow.pyfunc.model.PythonModelContext
            MLflow context (not used here but required by interface).
        model_input : pd.DataFrame
            DataFrame with columns:
            - user_id: int
            - n_movies_to_rec: int (optional)
            - new_user_interactions: list[int] (optional)

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns:
            - movie_ids: list[int]
            - movie_titles: list[str]
        """
        rows: list[dict[str, Any]] = []

        for _, row in model_input.iterrows():
            user_id: int = int(row["user_id"])
            n_rec: int = int(row.get("n_movies_to_rec", 5))
            new_inter: Optional[list[int]] = row.get("new_user_interactions", None)

            rec_ids: list[int] = recommend_item(
                als_model=self._state.model,
                data_csr=TRAIN_CSR_STORE.get_csr_matrix(),
                user_id=user_id,
                mappings=self._state.mappings,
                n_movies_to_rec=n_rec,
                new_user_interactions=new_inter,
                popular_item_ids=self._state.popular_item_ids,
            )

            movie_titles, movie_genres = get_movie_metadata(self._state.movie_id_dict, rec_ids)
            rows.append({
                "movie_ids": rec_ids,
                "movie_titles": movie_titles,
                "movie_genres": movie_genres,
            })

        return pd.DataFrame(rows)


def _get_champion_version(model_name: str) -> str | None:
    '''
    Given the model name this functions fetches the champion version and returns the model 
    version. 
    '''
    try:
        model_version = client.get_model_version_by_alias(name=model_name, alias="Champion")
        print("Found best model")
        return model_version.version
    except Exception as e:
        return None


def _get_run_id_for_version(model_name: str, version: str) -> str | None:
    '''
    Given the name of the trained model and it's version this function fetches the 
    run id corresponding to the run the given model was created with.
    '''
    try:
        return client.get_model_version(model_name, version).run_id
    except Exception:
        return None


def _load_champion_params(model_name: str) -> dict | None:
    """
    Tries to fetch hyperparams from Champion:
    1) best_params.json artifact on the Champion's run (preferred)
    2) otherwise from model version tags (bm25_K1, bm25_B, factors, reg, iters)
    Returns dict like:
      {"bm25_K1": 200, "bm25_B": 1.0, "factors": 256, "reg": 0.2, "iters": 25}
    or None if not available.
    """
    # Define return values
    champ_params = None
    champ_v = None
    rid = None
    
    # Load cham version and rid.
    champ_v = _get_champion_version(model_name)
    if champ_v:
        rid = _get_run_id_for_version(model_name, champ_v)

    # Load champ_params based on rid
    if rid:
        # 1) Try artifact best_params.json
        try:
            with tempfile.TemporaryDirectory() as tmpd:
                p = mlflow.artifacts.download_artifacts(
                    artifact_uri=f"runs:/{rid}/best_params.json",
                    dst_path=tmpd
                )
                with open(p, "r") as f:
                    bp = json.load(f)
                # normalize keys to the names you use in training below
                print("\nLoaded champ params from best_param.json")
                champ_params = {
                    "bm25_K1": bp.get("best_K1"),
                    "bm25_B":  bp.get("best_B"),
                    "factors": bp.get("best_factor"),
                    "reg":     bp.get("best_reg"),
                    "iters":   bp.get("best_iters"),
                }
        except Exception:
            pass
    
    # Load champ_params based on champ version
    if champ_params is None and champ_v:
        # 2) Fallback: read from version tags
        try:
            mv = client.get_model_version(model_name, champ_v)
            tags = mv.tags or {}
            def _f(key, cast):
                v = tags.get(key)
                return cast(v) if v is not None else None
            params = {
                "bm25_K1": _f("bm25_K1", int),
                "bm25_B":  _f("bm25_B", float),
                "factors": _f("factors", int),
                "reg":     _f("reg", float),
                "iters":   _f("iters", int),
            }
            if all(v is not None for v in params.values()):
                champ_params = params
        except Exception:
            pass

    return champ_params


# _________________________________________________________________________________________________________
# State holder
# _________________________________________________________________________________________________________

class BestParameters(BaseModel):
    '''Class to store the best model parameters.'''
    best_K1: int
    best_B: float
    best_factor: int
    best_reg: float
    best_iters: int


@dataclass
class Model_State:
    '''
    Holds the model state and everything needed to make recommendations with the champ model.
    '''
    model: AlternatingLeastSquares
    mappings: Mappings
    popular_item_ids: list[int]
    movie_id_dict: dict[int, str]


class TrainCSRStore:
    '''
    Infrastructure for working with the train_csr matrix of the champ model. The method 'save'
    saves the csr matrix to self.path abd stores it as npz file. 
    The load method loads the stored npz file and builds the matrix.
    '''
    def __init__(self, path: Path) -> None:
        self.path = path
        self.csr: Optional[csr_matrix] = None

    def load(self) -> None:
        '''
        Loads the npz csr matrix from self.path and stores it in self.csr.
        '''
        if self.path.exists():
            try:
                self.csr = load_npz(self.path).tocsr()
                print(f"[champ-store] Loaded champion train_csr from {self.path}")
            except Exception as e:
                print(f"[champ-store] Failed to load {self.path}: {e}")

    def save(self, mat: csr_matrix) -> None:
        '''
        Saves the given csr matrix into the path defined by self.path. 
        '''
        save_npz(self.path, mat, compressed=True)
        self.csr = mat
        print(f"[champ-store] Saved champion train_csr to {self.path}")


    def get_csr_matrix(self)-> Optional[csr_matrix]:
        '''
        Returns the csr matrix if already loaded, else None.
        '''
        if self.csr is not None:
            return self.csr
        else:
            try:
                self.load()
            except Exception as e:
                raise RuntimeError(
                f"[champ-store] Champion train_csr not available at {self.path}. "
                f"Has a champion been saved yet?"
            )
            return None


# Init the csr matrix store obj
TRAIN_CSR_STORE = TrainCSRStore(CHAMPION_TRAIN_CSR_PATH)


# _________________________________________________________________________________________________________
# Fast API schemas
# _________________________________________________________________________________________________________

class ALS_Parameter_Grid(BaseModel):
    '''
    Defines the structure of the grid parameters used to train the model.
    '''
    bm25_K1_list: Sequence[int] = Field(
        (100, 200),
        description="BM25 K1 values for document length normalization"
    )
    bm25_B_list: Sequence[float] = Field(
        (0.8, 1.0),
        description="BM25 B values for length normalization"
    )
    factors_list: Sequence[int] = Field(
        (128, 256),
        description="Number of latent factors in ALS"
    )
    reg_list: Sequence[float] = Field(
        (0.10, 0.20),
        description="Regularization parameter"
    )
    iters_list: Sequence[int] = Field(
        (25,),
        description="Number of ALS iterations"
    )


class TrainRequest(BaseModel):
    """Input schema for the `/train` endpoint."""
    n_users: int = Field(1000, description="Number of users to read (0 = full dataset)")
    pos_threshold: float = Field(4.0, description="Threshold for positive rating")

    als_parameter: ALS_Parameter_Grid = Field(
        ...,
        description=(
            "Grid of ALS hyperparameters (factors, reg, iters, K1, B) "
            "used for grid search."
        ),
    )
    n_popular_movies: int = Field(100, description="Number of popular movies for cold start functionality")
    

class TrainResponse(BaseModel):
    """Output schema for the `/train` endpoint."""
    best_param: BestParameters = Field(
        None,
        description="Best ALS hyperparameter combination found during training."
    )
    best_metrics: ALS_Metrics = Field(
        None,
        description="Metrics (precision@K, MAP@K) for the best parameter set."
    )


class RecommendRequest(BaseModel):
    """Input schema for the `/recommend` endpoint."""
    user_id: int = Field(
        ...,
        description="ID of the user for whom to generate recommendations.",
        example=42,
    )
    n_movies_to_rec: int = Field(
        5,
        gt=0,
        le=100,
        description="Number of movie recommendations to return (1–100).",
        example=10,
    )
    new_user_interactions: Optional[List[int]] = Field(
        None,
        description=(
            "Optional list of movie IDs recently watched or liked by the user. "
            "Used for cold-start or fold-in recommendations."
        ),
        example=[296, 318, 593],
    )


class RecommendResponse(BaseModel):
    """Output schema for the `/recommend` endpoint."""
    user_id: int = Field(..., description="User ID for which recommendations were generated.")
    movie_ids: List[int] = Field(..., description="List of recommended movie IDs sorted by relevance.")
    movie_titles: List[str] = Field(..., description="List of corresponding movie titles.")
    movie_genres: List[str] = Field(..., description="List of corresponding movie genres.")


# _________________________________________________________________________________________________________
# API Endpoints
# _________________________________________________________________________________________________________

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler: runs once at startup and once at shutdown.
    Ensures that the champ model and the corresponding train_csr matrix get's loaded
    when the API starts. 
    """
    TRAIN_CSR_STORE.load()                          # same logic as before
    print("[champ-store] CSR loaded at startup")

    # Load global champ model
    global CHAMP_MODEL
    # Init it directly with None because loading the model takes time and it needs to be available 
    # for the case that the Recommendation endpoint gets called.
    CHAMP_MODEL = None
    try:
        CHAMP_MODEL = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@Champion")
    except Exception as e:
        CHAMP_MODEL = None
    
    yield                                       # app runs while yielded
    print("[champ-store] App shutting down")    # optional cleanup
    # Optionally TRAIN_CSR_STORE.csr = None
    # or save to disk if needed


app = FastAPI(
    title="Movie Recommendation API",
    description="Movie recommendation system for training recommender model and make recommendation for users.",
    lifespan=lifespan
)

@app.get("/health", tags=["System"])
def health_check():
    """
    Lightweight healthcheck endpoint.
    Verifies connectivity to both the database and MLflow server.
    Returns 200 OK if both are reachable, else 500.
    """
    load_dotenv()
    DB_URL = os.getenv('DB_URL')
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    if not DB_URL:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection URL not found in environment variables."
        )
    status_report = {"timestamp": datetime.utcnow().isoformat()}
    
    # ✅ Check database connectivity
    if not DB_URL:
        status_report["database"] = "missing DB_URL env var"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        status_report["database"] = "reachable"
    except Exception as e:
        status_report["database"] = f"unreachable ({str(e)})"

    # ✅ Check MLflow connectivity
    if not MLFLOW_TRACKING_URI:
        status_report["mlflow"] = "missing MLFLOW_TRACKING_URI env var"
    try:
        mlflow_health_url = MLFLOW_TRACKING_URI.rstrip("/")
        response = requests.get(mlflow_health_url, timeout=5)
        if response.status_code == 200:
            status_report["mlflow"] = "reachable"
        else:
            status_report["mlflow"] = f"error ({response.status_code})"
    except Exception as e:
        status_report["mlflow"] = f"unreachable ({str(e)})"

    # ✅ Return aggregated report
    if (
        status_report.get("database") == "reachable"
        and status_report.get("mlflow") == "reachable"
    ):
        return status_report
    else:
        raise HTTPException(status_code=500, detail=status_report)
    

@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError):
    '''
    Every time a value error occurs Fast API routes this error to this handler instead of crashing.
    '''
    # e.g., "No positives after binarization" from prepare_data
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)},
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    '''
    Catches any other exception that wasn’t explicitly handled and returns a 500 JSON response instead.
    '''
    logging.exception("Unhandled error in /train: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error while training the model."},
    )

@app.post(
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


@app.post(
        "/train",
        response_model=TrainResponse,
        summary="Train or update the ALS recommendation model",
        description=(
        "Loads ratings + movies, prepares CSR matrices, runs a 3-stage advanced grid "
        "search to maximize MAP@K, compares the challenger to the current Champion, "
        "logs everything to MLflow, and updates the Champion if improved."
    ),
    response_description="Best hyperparameters and metrics found during training."
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
    # Simply store new model if old champ not won or not available
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
    
    # Updates the champ model functionality if new model is better than old champ model.
    update_champ_model(
        new_version=new_version,
        best_weighted_csr=best_weighted_csr
    )

    # Return train response
    return TrainResponse(best_param=best_param, best_metrics=metrics_ls[best_idx])



# Define recommendation endpoint
@app.post(
    "/recommend",
    response_model=RecommendResponse,
    summary="Get top-N movie recommendations for a user",
    description=(
        "Uses the in-memory Champion ALS model to recommend movies. "
        "Handles cold-start via fold-in if `new_user_interactions` are provided; "
        "otherwise falls back to popular items."
    ),
    response_description="Recommended movie IDs (and titles if available)."
)
def recommend_endpoint(recom_param: RecommendRequest):
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
    if CHAMP_MODEL is None:
        raise HTTPException(status_code=503, detail="Champion model not loaded yet")
    
    # Put request into df (Specified by MLFLow that model_input needs to be df or numpy.ndarray)
    df = pd.DataFrame([{
        "user_id": recom_param.user_id,
        "n_movies_to_rec": recom_param.n_movies_to_rec,
        "new_user_interactions": recom_param.new_user_interactions or []
    }])

    # Predict -> Movie recommendations
    df_movie_rec = CHAMP_MODEL.predict(df) 

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