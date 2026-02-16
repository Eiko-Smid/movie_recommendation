from typing import Dict, List, Tuple, Sequence, Optional, Any
import logging
from datetime import datetime
from time import sleep
from pathlib import Path
import os
import tempfile

from fastapi import FastAPI, Request, HTTPException, status

import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz
import json
from dataclasses import dataclass, asdict

import mlflow
from mlflow import MlflowClient
from mlflow.pyfunc import PythonModel
import joblib

from implicit.nearest_neighbours import bm25_weight
from implicit.als import AlternatingLeastSquares

from src.models.als_movie_rec import (
    Mappings,
    build_movie_id_dict,
    prepare_data,
    get_popular_items,
    ALS_Metrics,
    als_grid_search,
    recommend_item,
    get_movie_metadata,
)

from src.api.schemas import (
    TrainRequest,
    BestParameters
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

# CHAMP_MODEL: Optional[mlflow.pyfunc.PyFuncModel] = None

client = MlflowClient()

logger = logging.getLogger(__name__)

# _________________________________________________________________________________________________________
# State holder
# _________________________________________________________________________________________________________

@dataclass
class Model_State:
    '''
    Holds the model state and everything needed to make recommendations with the champ model.
    '''
    model: AlternatingLeastSquares
    mappings: Mappings
    popular_item_ids: list[int]
    movie_id_dict: dict[int, str]


# _________________________________________________________________________________________________________
# MLFLow helpers
# _________________________________________________________________________________________________________

def get_champion_model(request: Request):
    '''
    Reads the current champion model from the api app.state.
    '''
    model = getattr(request.app.state, "champion_model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not ready (no Champion model found). Train a model first.",
        )    
    return model


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
        app: FastAPI, 
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
    app.state.champion_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@Champion")


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
        # Read input row by row
        for _, row in model_input.iterrows():
            user_id: int = int(row["user_id"])
            n_rec: int = int(row.get("n_movies_to_rec", 5))
            new_inter: Optional[list[int]] = row.get("new_user_interactions", None)

            # Recommend an item for given user, n_rec's and user interactions (optional)
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


    def get_csr_matrix(self) -> Optional[csr_matrix]:
        """
        Returns the CSR matrix.

        Behavior:
        - If already loaded in memory -> return it.
        - If not loaded -> try loading from disk.
        - Log success or failure.
        """

        # Already in memory
        if self.csr is not None:
            logger.debug("[champ-store] CSR already loaded in memory.")
            return self.csr

        # Try loading
        try:
            self.load()

            if self.csr is not None:
                logger.info(f"[champ-store] CSR successfully loaded from {self.path}")
            else:
                logger.warning(f"[champ-store] CSR file not found at {self.path}")

        except Exception:
            logger.exception(f"[champ-store] Failed to load CSR from {self.path}")
            return None

        return self.csr
            


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


# Init the csr matrix store obj
TRAIN_CSR_STORE = TrainCSRStore(CHAMPION_TRAIN_CSR_PATH)
