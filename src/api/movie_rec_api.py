from __future__ import annotations
import logging
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from fastapi import FastAPI, HTTPException, status, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple, Mapping, Iterable, Any, TypedDict


import numpy as np
import pandas as pd

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
    get_movie_names
)


# _________________________________________________________________________________________________________
# Helper functions
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

    # Prepare rating data
    train_csr, test_csr, mappings = prepare_data(
        df=df_ratings,
        pos_threshold= train_param.pos_threshold,
    )

    # Compute item-movie dict for quick lookup
    movie_id_dict = build_movie_id_dict(df_movies)

    # get popular items for the case that a new user occurs that havent watched any movies by now.
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

    global model_state
    model_state = Model_State.init_from_training(
        model= best_model,
        train_csr=train_csr,
        test_csr=test_csr,
        mappings=mappings,
        movie_id_dict=movie_id_dict,
        popular_item_ids=popular_item_ids,
        best_params=best_param,
        best_metrics=metrics_ls[best_idx]
    )

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