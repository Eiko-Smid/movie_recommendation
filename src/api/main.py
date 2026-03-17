from __future__ import annotations

import logging
import os

import requests

os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"


import zlib
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text

import mlflow
from src.api.routers import admin, auth, rate_movie, recommend, train
from src.api.security import init_authorization

# Import sql request code
from src.db.database_session import engine
from src.models.management import (
    MODEL_NAME,
    TRAIN_CSR_STORE,
    get_model_version,
)
from src.observability.metrics import PrometheusHTTPMetricsMiddleware


def csr_fingerprint(X) -> str:
    h = 0
    for arr in (X.indptr, X.indices, X.data):
        h = zlib.crc32(arr.view(np.uint8), h)
    return f"{h & 0xFFFFFFFF:08x}"


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
    # Init the authorization
    init_authorization()

    # Load trained csr matrix
    TRAIN_CSR_STORE.load()
    logger.info("[startup] CSR loaded at startup")

    # Load global champ model
    logger.info(f"Model name is: {MODEL_NAME}")
    try:
        app.state.champion_model = mlflow.pyfunc.load_model(
            f"models:/{MODEL_NAME}@Champion"
        )
        app.state.champion_model_version = get_model_version(model_name=MODEL_NAME)
        logger.info("[startup] Stored champ model in app.state.champion_model")
        logger.info(
            "[startup] Stored champ model version in app.state.champion_model_version"
        )
    except Exception:
        app.state.champion_model = None
        app.state.champion_model_version = None
        logger.exception(
            "[startup] Failed to load champion model or champ model version from MLflow"
        )

    yield  # app runs while yielded
    print("[champ-store] App shutting down")  # optional cleanup
    app.state.champion_model = None
    # Optionally TRAIN_CSR_STORE.csr = None
    # or save to disk if needed


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


@app.get("/health", tags=["System"])
def health_check():
    """
    Lightweight healthcheck endpoint.
    Verifies connectivity to both the database and MLflow server.
    Returns 200 OK if both are reachable, else 500.
    """
    load_dotenv()
    DB_URL = os.getenv("DB_URL")
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    if not DB_URL:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection URL not found in environment variables.",
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
