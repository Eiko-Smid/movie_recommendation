from __future__ import annotations

import logging
import os

import requests

os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse, Response
from mlflow.exceptions import RestException
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text
from sqlalchemy.orm import Session

import mlflow

# Import AppState
from src.api.app_state import AppState
from src.api.routers import admin, auth, rate_movie, recommend, train, analytics
from src.api.security import init_authorization

# Import sql request code
from src.db.database_session import get_db
from src.models.management import (
    MODEL_NAME,
    TRAIN_CSR_STORE,
    get_model_version,
)
from src.observability.metrics import PrometheusHTTPMetricsMiddleware

# _________________________________________________________________________________________________________
# API Endpoints
# _________________________________________________________________________________________________________

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler: runs once at startup and once at shutdown.
    Ensures that the champ model and the corresponding train_csr matrix get's loaded
    when the API starts.
    """
    # Load testing var from env, default to false if not set
    TESTING = os.getenv("TESTING", "false").lower() == "true"

    # procedure for non-testing environment
    if not TESTING:    
        # Init the authorization
        init_authorization()

        # Load trained csr matrix
        TRAIN_CSR_STORE.load()
        logger.info("[startup] CSR loaded at startup")

        # Load global champ model
        logger.info(f"Model name is: {MODEL_NAME}")
        try:
            # Load current champ model and model version from mlflow and store in app state
            champ_model = mlflow.pyfunc.load_model(
                f"models:/{MODEL_NAME}@Champion"
            )
            champ_model_version = get_model_version(model_name=MODEL_NAME)
            app.state.app_state = AppState(
                champ_model=champ_model,
                champ_model_version=champ_model_version
            )
            logger.info("[startup] Stored champ model and version in app.state.app_state")
        
        # Accept if model is empty at start time.
        except RestException as e:
            # expected case: no Champion yet
            logger.warning(f"[startup] No Champion model found in MLflow yet.\nDetails:\n{e}")
            app.state.app_state = AppState(
                champ_model=None,
                champ_model_version=None
            )
        # handle real rpoblem
        except Exception as e:
            logger.error("[startup] Critical error while loading model")
            raise e  
    else:
        # Init app.state with None values during CI check, as we don't have a model or a model version during check
        logger.info("[startup] CSR not loaded during CI check")
        logger.info("Model name is: Test_Model_CI_check")
        app.state.app_state = AppState(
            champ_model=None,
            champ_model_version=None
        )
        logger.info("[startup] Stored None in app.state.app_state during CI check")
        logger.info("[startup] Stored None in app.state.app_state.champ_model_version during CI check")
        
    # Wait for api to shut down
    yield  
    # Cleanup after shutdown
    print("[champ-store] App shutting down")
    app.state.app_state = AppState(
        champ_model=None,
        champ_model_version=None
    )



app = FastAPI(
    title="Movie Recommendation API",
    description="Movie recommendation system for training recommender model and make recommendation for users.",
    lifespan=lifespan,
)

# Register Prometheus HTTP metrics middleware
# THis enables the automatic tracking of all endpoints, according to the middleware
app.add_middleware(PrometheusHTTPMetricsMiddleware)

# Include router endpoints
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(train.router)
app.include_router(recommend.router)
app.include_router(rate_movie.router)
app.include_router(analytics.router)


@app.get("/health", tags=["System"])
def health():
    '''
    Lightweight health check to ensure api is running.
    '''
    return {"status": "ok"}


@app.get("/health/full", tags=["System"])
def health_advanced(db: Session = Depends(get_db)):
    '''
    Performs an extended health check for the API and its external dependencies.

    This endpoint verifies that the application can communicate with the configured
    database and the MLflow tracking server. It returns a structured JSON response
    with the health status of each dependency and an overall HTTP status code:
    200 if both services are available, otherwise 500.

    Parameters
    ----------
    db : Session, optional
        SQLAlchemy database session injected by FastAPI via dependency injection.
        Used to execute a lightweight query in order to verify database connectivity.

    Returns
    -------
    JSONResponse
        A JSON response containing the health state of the database and MLflow
        connection in the form:

        {
            "DB": {
                "ok": bool,
                "message": str
            },
            "MLflow": {
                "ok": bool,
                "message": str
            }
        }

    The response status code is 200 if both checks succeed, otherwise 500.
    '''
    # Define status vals and msgs
    db_status = False
    db_status_msg = ""
    mlflow_status = False
    mlflow_status_msg = ""

    # Test DB connection
    try:
        # Get first element
        db.execute(text("SELECT 1")).scalar()
        db_status = True
        db_status_msg = "DB connection healthy."
    except Exception as e:
        db_status = False
        db_status_msg = "DB connection failed."
        logger.error(f"DB health echeck failed: {e}")

    # Try get mlfow tracking uri from env
    try:
        # Get mlflow tracking uri from env, if not set raise exception
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not mlflow_tracking_uri:
            raise ValueError("MLFLOW_TRACKING_URI not set")
    except Exception as e:
        # If env var is not set, set mlflow status to unhealthy and capture error message
        mlflow_tracking_uri = None
        mlflow_status = False
        mlflow_status_msg = "No mlfow tracking uri found in env vars. Check if exists and if name is correct."
        logger.error(f"No mlfow tracking uri found: {e}")

    # Test MLflow connection
    if mlflow_tracking_uri:
        try:
            # Request mlflow server
            mlflow_url = mlflow_tracking_uri.rstrip("/")
            response = requests.get(mlflow_url, timeout=2)
            response.raise_for_status()
            # If request is successful, set mlflow status to healthy
            mlflow_status = True
            mlflow_status_msg = "MLflow connection healthy."
        except Exception as e:
            # If request fails, set mlflow status to unhealthy and capture error message
            mlflow_status = False
            mlflow_status_msg = "Mlfow connection couldn't be established."
            logger.error(f"No mlflow connection: {e}")

    # Check overall health
    healthy = db_status and mlflow_status

    return JSONResponse(
        status_code=status.HTTP_200_OK
        if healthy
        else status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "DB": {
                "ok": db_status,
                "message": db_status_msg,
            },
            "MLflow": {
                "ok": mlflow_status,
                "message": mlflow_status_msg,
            },
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError):
    """
    Every time a value error occurs Fast API routes this error to this handler instead of crashing.
    """
    # e.g., "No positives after binarization" from prepare_data
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    """
    Catches any other exception that wasn’t explicitly handled and returns a 500 JSON response instead.
    """
    logging.exception("Unhandled error in /train: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error while training the model."},
    )


@app.get("/metrics")
def metrics():
    """
    Prometheus scrape endpoint.
    Returns all registered metrics in Prometheus text format.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
