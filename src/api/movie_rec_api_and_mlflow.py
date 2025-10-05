from __future__ import annotations
import logging
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from pathlib import Path

from fastapi import FastAPI, HTTPException, status, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple, Mapping, Iterable, Any, TypedDict

from datetime import datetime

import numpy as np
import pandas as pd

import json
import hashlib
import platform

import mlflow
from mlflow import MlflowClient
import joblib
from mlflow.pyfunc import PythonModel

from scipy.sparse import coo_matrix, csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight

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



# _________________________________________________________________________________________________________
# Helper functions and classes 
# _________________________________________________________________________________________________________


def _model_ready() -> bool:
    '''
    Helper functions that checks if the model is ready to recommend a movie.
    '''
    return (
        model_state is not None and
        model_state.model is not None and
        model_state.full_csr_weighted is not None and
        model_state.mappings is not None and
        model_state.movie_id_dict is not None
    )

def _sha256(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "unavailable"


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
                data_csr=self._state.full_csr_weighted,
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
    '''Class to store the information of the current model. This includes the model itself,
    the datatset as well ans mappings and best parameters + metrics.'''
    model: Optional[AlternatingLeastSquares] = None
    train_csr: Optional[csr_matrix] = None
    test_csr: Optional[csr_matrix] = None
    full_csr_weighted: Optional[csr_matrix] = None
    mappings: Optional[Mappings] = None
    movie_id_dict: Optional[Dict[int, str]] = None
    popular_item_ids: Optional[List[int]] = None
    best_metrics: Optional[ALS_Metrics] = None
    best_parameters: Optional[BestParameters] = None


    def build_full_csr(self) -> None:
        assert self.train_csr is not None and self.test_csr is not None, "train/test missing"
        assert self.best_parameters is not None, "best params missing"
        full_csr = self.train_csr + self.test_csr
        self.full_csr_weighted = bm25_weight(
            full_csr,
            K1=self.best_parameters.best_K1,
            B=self.best_parameters.best_B
        ).tocsr()


    @classmethod
    def init_from_training(
        cls,
        *,
        model: AlternatingLeastSquares,
        train_csr: csr_matrix,
        test_csr: csr_matrix,
        mappings: Mappings,
        movie_id_dict: Dict[int, str],
        popular_item_ids: List[int],
        best_params: BestParameters,
        best_metrics: ALS_Metrics,
    ) -> "Model_State":
        state = cls(
            model=model,
            train_csr=train_csr,
            test_csr=test_csr,
            mappings=mappings,
            movie_id_dict=movie_id_dict,
            popular_item_ids=popular_item_ids,
            best_parameters=best_params,
            best_metrics=best_metrics,
        )
        state.build_full_csr()
        return state
    

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
# Global API Exception handler 
# _________________________________________________________________________________________________________

# Create API obj
app = FastAPI(title="ALS Movie Recommendation", version="0.1.0")

# Create model_state place holder. Will be recreated in the training endpoint but is needed before
model_state: Optional[Model_State] = None

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



# _________________________________________________________________________________________________________
# API Endpoints
# _________________________________________________________________________________________________________

@app.get("/health")
def health():
    '''
    API endpoint that checks if the API is still alive and if model_state is already initialized.
    '''
    return {"ok": True, "model_ready": model_state is not None}


@app.post("/train", response_model=TrainResponse)
def train_endpoint(train_param: TrainRequest):
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

    # Set up MLflow experiment + start parent run
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "als_movielens_20m")
    mlflow.set_experiment(exp_name)
    
    run_name = f"train | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | n_rows={n_rows}"
    with mlflow.start_run(run_name=run_name) as run:
        # Tags
        mlflow.set_tags({
            "component": "train_api",
            "framework": "implicit",
            "algorithm": "ALS",
            "host": platform.node(),
            "os": platform.platform()
        })

        # Save parameter about data and search space.
        mlflow.log_params({
            "data_path_ratings": data_path_ratings,
            "data_path_movies": data_path_movies,
            "n_rows": n_rows,
            "pos_threshold": train_param.pos_threshold,
            "ratings_sha256": _sha256(data_path_ratings),
            "movies_sha256": _sha256(data_path_movies)
        })

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

        # Prepare rating data
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

        # Grid search
        _ , metrics_ls, parameter_ls, best_idx = als_grid_search(
            train_csr=train_csr,
            test_csr=test_csr,
            bm25_K1_list=train_param.als_parameter.bm25_K1_list,
            bm25_B_list=train_param.als_parameter.bm25_B_list,
            factors_list=train_param.als_parameter.factors_list,
            reg_list=train_param.als_parameter.reg_list,
            iters_list=train_param.als_parameter.iters_list
        )

        # Log all grid results as one csv artifact
        try:
            rows = []
            for params, m in zip(parameter_ls, metrics_ls):
                # TODO: Change K to an actual parameter value! -> Define parameter
                rows.append({
                    **params,
                    "precision_@_k": float(m.prec_at_k),
                    "map_at_k": float(m.map_at_k)
                })
            # Create df from grid params and corresponding metrics
            grid_df = pd.DataFrame(rows)
            # Create storage path 
            artifacts_dir = Path("mlflow_artifacts")
            artifacts_dir.mkdir(exist_ok=True)
            grid_path = artifacts_dir / "grid_results.csv"
            # Store df as csv file
            grid_df.to_csv(grid_path, index=False)
            mlflow.log_artifact(str(grid_path), artifact_path="search")
        except Exception as e:
            logging.warning(f"Could not log grid results artifact: {e}")
                
        # Retrain model on whole data
        # Build complete data in csr 
        complete_data_csr = train_csr + test_csr
        
        # Get best params
        best_K1 = parameter_ls[best_idx]["bm25_K1"]
        best_B = parameter_ls[best_idx]["bm25_B"]
        best_factors = parameter_ls[best_idx]["factors"]
        best_reg = parameter_ls[best_idx]["reg"]
        best_iters = parameter_ls[best_idx]["iters"]

        # Weight complete data
        complete_data_weighted_csr = bm25_weight(complete_data_csr, K1=best_K1, B=best_B).tocsr()

        # Define model with best params
        best_model = AlternatingLeastSquares(
            factors=best_factors,
            regularization=best_reg,
            iterations=best_iters
        )
        print("\nTrain model on whole dataset with best params...")
        best_model.fit(complete_data_weighted_csr)

        # Store model data
        best_param = BestParameters(
                best_K1=best_K1,
                best_B=best_B,
                best_factor=best_factors,
                best_reg=best_reg,
                best_iters=best_iters,
            )

        best_metrics = metrics_ls[best_idx]
        global model_state
        model_state = Model_State.init_from_training(
            model= best_model,
            train_csr=train_csr,
            test_csr=test_csr,
            mappings=mappings,
            movie_id_dict=movie_id_dict,
            popular_item_ids=popular_item_ids,
            best_params=best_param,
            best_metrics=best_metrics
        )

        # Store best metrics in mlflow
        mlflow.log_metrics({
            "best_precision_at_k": float(best_metrics.prec_at_k),
            "best_map_at_k": float(best_metrics.map_at_k)
        })
        mlflow.log_dict(asdict(best_param), "best_params.json")

        # _____________________________________________________________________
        # Log a pyfunc model
        # _____________________________________________________________________
        
        # Save trained model with jolib under dir "mlflow_artifacts"
        artifacts_dir = Path("mlflow_artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        state_pkl = artifacts_dir / "model_state.pkl"
        joblib.dump(model_state, state_pkl)

        # Define a inoput example for using the model saved as mlflow artifact.
        input_example = pd.DataFrame([{
            "user_id": 1,
            "n_movies_to_rec": 5.0,
            "new_user_interactions": [1, 2, 3]
        }])

        # Define the output of the _model.predict() method from "ALSRecommenderPyFunc"
        output_example = pd.DataFrame([{
            "movie_ids": [1, 2],
            "movie_titles": ["Movie A", "Movie B"]
        }])
        signature = mlflow.models.infer_signature(
            input_example.astype({"n_movies_to_rec": "float64"}),
            output_example)

        model_info = mlflow.pyfunc.log_model(
            # artifact_path="model",
            name="als_pyfunc",
            python_model=ALSRecommenderPyFunc(),
            artifacts={"state_path": str(state_pkl)},
            signature=signature,
            input_example=input_example,
            # Defines requirements needed to run hte model.
            pip_requirements=[
                "mlflow",
                "pandas",
                "numpy",
                "scipy",
                "implicit",
                "joblib",
            ],
        )
        
        # _____________________________________________________________________
        # Register model version
        # _____________________________________________________________________
        model_name = os.getenv("MLFLOW_MODEL_NAME", "ALSRecommender")
        run_id = mlflow.active_run().info.run_id
        
        registered = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=model_name
        )
        
        # Tag the run with the created version for convenience
        mlflow.set_tags({"registered_model": model_name, "registered_version": registered.version})


    # Return train response
    return TrainResponse(best_param=best_param, best_metrics=metrics_ls[best_idx])


@app.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(recom_param: RecommendRequest):

    # Check if model is ready
    if not _model_ready():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model not trained yes. Call /train endpoint first."
        )

    # Recommend movies for a given user
    try:
        rec_movie_ids = recommend_item(
            als_model=model_state.model,
            data_csr=model_state.full_csr_weighted,
            user_id=recom_param.user_id,
            mappings=model_state.mappings,
            n_movies_to_rec=recom_param.n_movies_to_rec,
            new_user_interactions=recom_param.new_user_interactions,
            popular_item_ids=model_state.popular_item_ids
        )
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Ivalid ID mapping: {e}"
        )
    except Exception as e:
        # catch-all safety net
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute recommendations: {e}"
        )

    # Transform movie ids to movie names
    rec_movie_names = get_movie_names(movie_id_dict=model_state.movie_id_dict, movie_ids=rec_movie_ids)

    # Return Response
    return RecommendResponse(
        user_id=recom_param.user_id,
        movie_ids=rec_movie_ids,
        movie_titles=rec_movie_names
    )