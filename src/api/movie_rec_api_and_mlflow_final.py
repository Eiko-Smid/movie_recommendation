from __future__ import annotations
import logging
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from pathlib import Path
import tempfile

from fastapi import FastAPI, HTTPException, status, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple, Mapping, Iterable, Any, TypedDict, Callable

from datetime import datetime

import numpy as np
import pandas as pd

import json
import hashlib
import platform

import mlflow
from mlflow import MlflowClient
# from mlflow.tracking import MlflowClient
import joblib
from mlflow.pyfunc import PythonModel

from scipy.sparse import coo_matrix, csr_matrix, save_npz, load_npz
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight

from time import time, sleep

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
    evaluate_als
)




'''
TODOs:

1) 

I think the current version has the bug that it includes movies that the user might have seen now but
not at the point were the champ model has been trained.

'''

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

def load_data(train_param: TrainRequest):
    # Definitions
    data_path_ratings = "data/ml-20m/ratings.csv"
    data_path_movies = "data/ml-20m/movies.csv"

    # Load data
    n_rows = train_param.n_rows
    # Check for existing paths
    if not os.path.exists(data_path_ratings):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ratings csv file not found. Path is:\n{data_path_ratings}"
        )

    if not os.path.exists(data_path_movies):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Movies csv file not found. Path is:\n{data_path_movies}"
        )
    
    # Try to load data
    try:
        df_ratings = pd.read_csv(data_path_ratings, nrows= n_rows if n_rows > 0 else None)
        df_movies = pd.read_csv(data_path_movies)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read CSVs: {e}"
            )
    
    return df_ratings, df_movies



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
                data_csr=self._state.train_csr,
                user_id=user_id,
                mappings=self._state.mappings,
                n_movies_to_rec=n_rec,
                new_user_interactions=new_inter,
                popular_item_ids=self._state.popular_item_ids,
            )

            rows.append({
                "movie_ids": rec_ids,
                "movie_titles": get_movie_names(self._state.movie_id_dict, rec_ids),
            })

        return pd.DataFrame(rows)


def _get_champion_version(model_name: str) -> str | None:
    try:
        model_version = client.get_model_version_by_alias(name=model_name, alias="Champion")
        print("Found best model")
        return model_version.version
    except Exception as e:
        return None


def _get_run_id_for_version(model_name: str, version: str) -> str | None:
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
    # TODO: We got 6 returns in this function -> Replace this with return var and one return
    champ_v = _get_champion_version(model_name)
    if not champ_v:
        return None

    rid = _get_run_id_for_version(model_name, champ_v)
    if not rid:
        return None

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
            return {
                "bm25_K1": bp.get("best_K1"),
                "bm25_B":  bp.get("best_B"),
                "factors": bp.get("best_factor"),
                "reg":     bp.get("best_reg"),
                "iters":   bp.get("best_iters"),
            }
    except Exception:
        pass

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
            return params
    except Exception:
        pass

    return None


def _build_user_row_from_item_ids(
    item_ids: Sequence[int],
    item_id_to_index: dict[int, int],
    n_items: int,
    dtype="float32",
    verbose: bool = False
) -> csr_matrix:
    """Safely create a 1×n_items sparse row, ignoring unknown movieIds."""
    valid_cols = []
    unknown_items = []
    for mid in item_ids:
        if mid in item_id_to_index:
            valid_cols.append(item_id_to_index[mid])
        else:
            unknown_items.append(mid)
    if verbose and unknown_items:
        print(f"[WARN] Ignoring {len(unknown_items)} unknown movies: {unknown_items[:5]}...")
    if not valid_cols:
        return csr_matrix((1, n_items), dtype=dtype)
    data = [1.0] * len(valid_cols)
    return coo_matrix((data, ([0]*len(valid_cols), valid_cols)), shape=(1, n_items), dtype=dtype).tocsr()


def recommend_items_stateless(
    als_model: AlternatingLeastSquares,
    mappings: Mappings,
    n_movies_to_rec: int = 5,
    *,
    user_id: Optional[int] = None,
    interactions_item_ids: Optional[Sequence[int]] = None,
    fetch_user_interactions: Optional[Callable[[int], Sequence[int]]] = None,
    popular_item_ids: Optional[Sequence[int]] = None,
    fold_in_always: bool = True
) -> List[int]:    
    """
    Recommend movies for a user using a stateless approach (no train_csr required).

    Parameters
    ----------
    als_model : AlternatingLeastSquares
        Trained implicit ALS model.
    mappings : Mappings
        Object containing ID↔index mappings for users and items.
    n_movies_to_rec : int, default=5
        Number of movie recommendations to return.
    user_id : Optional[int], default=None
        ID of the user for whom recommendations are generated.
    interactions_item_ids : Optional[Sequence[int]], default=None
        List of movieIds the user recently interacted with (positive signals).
    fetch_user_interactions : Optional[Callable[[int], Sequence[int]]], default=None
        Callback function to fetch a user's latest movie interactions if not provided.
    popular_item_ids : Optional[Sequence[int]], default=None
        List of popular item IDs for cold-start fallback.
    fold_in_always : bool, default=True
        If True, always perform fold-in (recalculate_user=True), even for known users.

    Returns
    -------
    List[int]
        List of recommended movieIds (original IDs, not internal indices).
    """

    n_items = len(mappings.item_index_to_id)

    # 1. Fetch or receive latest interactions
    if interactions_item_ids is None and user_id is not None and fetch_user_interactions:
        interactions_item_ids = fetch_user_interactions(user_id) or []

    # 2. Build sparse row safely
    user_row = _build_user_row_from_item_ids(
        interactions_item_ids or [], mappings.item_id_to_index, n_items
    )

    # Helper for reverse mapping
    def _map_items_back(item_idxs):
        return [mappings.item_index_to_id[i] for i in item_idxs]

    # 3. If we have any valid known items, fold-in recommendation
    if user_row.nnz > 0:
        rec_items, _ = als_model.recommend(
            user=0 if fold_in_always else mappings.user_id_to_index.get(user_id, 0),
            user_items=user_row,
            N=n_movies_to_rec,
            filter_already_liked_items=True,
            recalculate_user=True
        )
        return _map_items_back(rec_items)

    # 4. If all interactions were new / unknown movies
    if interactions_item_ids:
        print("[INFO] All provided movieIds are unknown to the Champion model — returning popular items.")
        if popular_item_ids:
            return list(popular_item_ids[:n_movies_to_rec])
        return []

    # 5. Known user but no interactions
    if user_id is not None and user_id in mappings.user_id_to_index:
        uidx = mappings.user_id_to_index[user_id]
        empty_row = csr_matrix((1, n_items), dtype="float32")
        rec_items, _ = als_model.recommend(
            user=uidx, user_items=empty_row, N=n_movies_to_rec,
            filter_already_liked_items=False, recalculate_user=False
        )
        return _map_items_back(rec_items)

    # 6. Final cold-start fallback
    if popular_item_ids:
        return list(popular_item_ids[:n_movies_to_rec])

    return []

# _________________________________________________________________________________________________________
# State holder
# _________________________________________________________________________________________________________

@dataclass
class BestParameters:
    best_K1: int
    best_B: float
    best_factor: int
    best_reg: float
    best_iters: int


@dataclass
class Model_State:
    model: AlternatingLeastSquares
    train_csr: csr_matrix
    mappings: Mappings
    popular_item_ids: list[int]
    movie_id_dict: dict[int, str]


class TrainCSRStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.csr: Optional[csr_matrix] = None

    def load(self) -> None:
        if self.path.exists():
            try:
                self.csr = load_npz(self.path).tocsr()
                print(f"[champ-store] Loaded champion train_csr from {self.path}")
            except Exception as e:
                print(f"[champ-store] Failed to load {self.path}: {e}")

    def save(self, mat: csr_matrix) -> None:
        save_npz(self.path, mat, compressed=True)
        self.csr = mat
        print(f"[champ-store] Saved champion train_csr to {self.path}")


TRAIN_CSR_STORE = TrainCSRStore(CHAMPION_TRAIN_CSR_PATH)


# _________________________________________________________________________________________________________
# Fast API schemas
# _________________________________________________________________________________________________________

class ALS_Parameter_Grid(BaseModel):
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
    n_rows: int = Field(10000, description="Number of rows to read (0 = full dataset)")
    pos_threshold: float = Field(4.0, description="Threshold for positive rating")
    als_parameter: ALS_Parameter_Grid

    # user_to_recommend: int = Field(0, description="User ID to generate recommendations for")
    n_popular_movies: int = Field(100, description="Number of popular movies for cold start functionality")
    # n_movies_to_rec: int = Field(5, description="Number of movies to recommend")


class TrainResponse(BaseModel):
    best_param: BestParameters = None
    best_metrics : ALS_Metrics = None


class RecommendRequest(BaseModel):
    user_id: int
    n_movies_to_rec: int = Field(5, gt=0, le=100)
    new_user_interactions: Optional[List[int]] = None

class RecommendResponse(BaseModel):
    user_id: int
    movie_ids: List[int]
    movie_titles: List[str]



# _________________________________________________________________________________________________________
# API Endpoints
# _________________________________________________________________________________________________________

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler: runs once at startup and once at shutdown."""
    TRAIN_CSR_STORE.load()            # same logic as before
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
    
    yield                             # app runs while yielded
    print("[champ-store] App shutting down")  # optional cleanup
    # Optionally TRAIN_CSR_STORE.csr = None
    # or save to disk if needed


app = FastAPI(title="Retrain + MLflow (simple)", lifespan=lifespan)


@app.post("/train", response_model=TrainResponse)
def train_endpoint(train_param: TrainRequest):
    # Load data
    df_ratings, df_movies = load_data(train_param=train_param)

    # Only use random subset of df
    perc = 0.8
    n_samples = int(df_ratings.shape[0] * 0.8)
    df_ratings = df_ratings.sample(n=n_samples, random_state=np.random.randint(0, 1_000_000))

    # Set up model training
    # Prepare data
    train_csr, test_csr, mappings = prepare_data(
        df=df_ratings,
        pos_threshold= train_param.pos_threshold,
    )

    # Compute item-movie dict for quick lookup
    movie_id_dict = build_movie_id_dict(df_movies)

    # Get popular items for the case that a new user occurs that hasn't watched any movies by now.
    popular_item_ids = get_popular_items(
        df=df_ratings,
        top_n=train_param.n_popular_movies,
        threshold=train_param.pos_threshold)
    print(f"\nShape of popular_item_ids is {len(popular_item_ids)}")


    # Train model
    # Grid search
    # TODO: Adapt the return param of grid search such that the element type of the metrics are known
    model , metrics_ls, parameter_ls, best_idx = als_grid_search(
        train_csr=train_csr,
        test_csr=test_csr,
        bm25_K1_list=train_param.als_parameter.bm25_K1_list,
        bm25_B_list=train_param.als_parameter.bm25_B_list,
        factors_list=train_param.als_parameter.factors_list,
        reg_list=train_param.als_parameter.reg_list,
        iters_list=train_param.als_parameter.iters_list
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


    # Define run name beased on date-time
    run_name = f"train | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | n_rows={train_param.n_rows}"
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

        # Log best params from grid search
        mlflow.log_dict(asdict(best_param), "best_params.json")

        # Log metrics of model training
        mlflow.log_metrics({
            "prec_at_k":float(best_metrics.prec_at_k),
            "map_at_k":float(best_metrics.map_at_k)
        })

        # Create path to store model artifacts. The pyfunc model will need this later on!
        artifacts_dir = Path("artifacts_tmp")
        artifacts_dir.mkdir(exist_ok=True)     
        artifacts_path = artifacts_dir / "model_state.joblib"
        
        # Store model with joblib
        # joblib.dump(model, model_path)

       # Recompute the BM25-weighted matrix that matches the *best* params
        best_weighted_csr = bm25_weight(train_csr, K1=best_param.best_K1, B=best_param.best_B).tocsr()

        # Build a full state (so colleagues loading via MLflow get an immediately-usable model)
        state_obj = Model_State(
            model=model,
            train_csr=best_weighted_csr,
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

    # Load best params from champ version
    champ_params = _load_champion_params(model_name=MODEL_NAME)

    if champ_params is None:
        # Set model improved flag to true if not champ model currently exists 
        champ_map = None
        improved = True
    else:
        # Retrain champ model on new data using the old best params -> Only one training
        champ_model_, champ_metrics_, champ_params_, champ_idx_ = als_grid_search(
            train_csr=train_csr,
            test_csr=test_csr,
            bm25_K1_list=[champ_params["bm25_K1"]],
            bm25_B_list=[champ_params["bm25_B"]],
            factors_list=[champ_params["factors"]],
            reg_list=[champ_params["reg"]],
            iters_list=[champ_params["iters"]]
        )
        # champ_prec = champ_metrics[champ_idx].prec_at_k
        champ_map = champ_metrics_[champ_idx_].map_at_k
        # model_prec = best_metrics.prec_at_k
        model_map = best_metrics.map_at_k

        # Decide new model is better than old champ model
        if model_map > champ_map:
            improved = True
        else: 
            improved = False
    
    if improved:
        # Mark new model as Champ model
        client.set_registered_model_alias(MODEL_NAME, "Champion", new_version)

        # Save the csr matrix that has been used to train the new champ model -> Predictions enabled
        TRAIN_CSR_STORE.save(best_weighted_csr)

        # Load the new champ model
        global CHAMP_MODEL
        CHAMP_MODEL = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@Champion")

    # Return train response
    return TrainResponse(best_param=best_param, best_metrics=metrics_ls[best_idx])



# Define recommendation endpoint
@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(recom_param: RecommendRequest):
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
    movie_titles = df_movie_rec.iloc[0].get("movie_titles", None)

    return RecommendResponse(
        user_id=recom_param.user_id,
        movie_ids=movie_ids,
        movie_titles=movie_titles
    )