"""
src/observability/http_metrics.py

Automatic HTTP metrics for FastAPI (Prometheus).

What this module does
---------------------
This module provides a Starlette middleware that automatically records
HTTP metrics for *all* FastAPI endpoints, without manual .inc() calls in routers.

Recorded metrics
----------------
1) http_requests_total (Counter)
   - Counts requests by method, route template, and status code.

2) http_request_duration_seconds (Histogram)
   - Records request latency in seconds by method and route template.

3) http_requests_in_flight (Gauge)
   - Tracks current in-flight requests by method and route template.

Why route templates?
--------------------
We label using FastAPI route templates like:
  "/admin/users/{user_id}/role"
instead of raw URLs like:
  "/admin/users/123/role"
This avoids high-cardinality metrics (Prometheus performance killer).

Special labels
--------------
If a request cannot be matched to a route template:
- 404 -> path="__not_found__"
- otherwise -> path="__unmatched__"
"""

from __future__ import annotations

import time
from typing import Optional, Set

from prometheus_client import Counter, Histogram, Gauge
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


# =============================================================================
# Prometheus metric definitions (created once per Python process)
# =============================================================================
"""
Global Prometheus counter for recommendation requests.
Labels:
    endpoint: Name of endpoint
    result: success or failure
"""

RECOMMENDATION_REQUEST_COUNT = Counter(
    name="reco_requests_total",
    documentation="Total number of recommendation requests.",
    labelnames=["endpoint", "result"],
)

RECOMMENDATION_REQUEST_DURATIONS_SEC = Histogram(
    name="reco_request_duration_sec",
    documentation="Duration time for a movie recommendation in seconds,",
)

DATA_NUMB_TRAIN_USERS = Gauge(
    name="data_numb_train_users",
    documentation="Number of unique users in the training ratings data.",
)

DATA_NUMB_TRAIN_INTERACTIONS = Gauge(
    name="data_numb_train_interactions",
    documentation="Number of user-item interactions in the training ratings data.",
)

DATA_NUMB_TEST_USERS = Gauge(
    name="data_numb_test_users",
    documentation="Number of unique users in the test ratings data.",
)

DATA_NUMB_TEST_INTERACTIONS = Gauge(
    name="data_numb_test_interactions",
    documentation="Number of user-item interactions in the test ratings data.",
)

MODEL_DATA_LOADING_DURATION_SEC = Histogram(
    name="model_data_loading_duration_sec",
    documentation=(
        "Duration time for loading training ratings data and user ratings data," 
        "combine them and filter them by materialized view."
    ),
)

MODEL_PREPROCESSING_DURATIONS_SEC = Histogram(
    name="model_preprocessing_duration_sec",
    documentation="Duration time for model preprocessing in seconds.",
)

MODEL_TRAINING_DURATIONS_SEC = Histogram(
    name="model_training_duration_sec",
    documentation="Duration time for model training in seconds.",
)

MODEL_PREC_AT_K = Gauge(
    name="model_precision_at_k",
    documentation="Precision at K of the currently production model.",
    labelnames=["model_version", "K"],
)

MODEL_MAP_AT_K = Gauge(
    name="model_map_at_k",
    documentation="MAP at K of the currently production champion model.",
    labelnames=["model_version", "K"],
)

MODEL_SERVED_TOTAL = Counter(
    name="model_served_total",
    documentation="Number of times the champion model served predictions.",
    labelnames=["model_version"],
)

"""
Counter: total HTTP request count.

Labels:
    method: HTTP method (GET, POST, ...)
    path: route template (e.g. "/health" or "/admin/users/{user_id}/role")
    status: HTTP status code string (e.g. "200", "401", "500")
"""
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "path", "status"],
)

"""
Histogram: HTTP request duration in seconds.

Labels:
    method: HTTP method
    path: route template
"""
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

"""
Gauge: number of in-flight HTTP requests (currently being processed).

Labels:
    method: HTTP method
    path: route template
"""
HTTP_REQUESTS_IN_FLIGHT = Gauge(
    "http_requests_in_flight",
    "Number of HTTP requests in flight",
    ["method"],
)

# List of metrics to exclude from http track
EXCLUDE_TEMPPLATES: Set = {
    "/metrics",
}


# =============================================================================
# Helper functions
# =============================================================================

def get_endpoint_route(request: Request) -> str:
    """
    Determine endpoint route of endpoint that corresponds to given request.
    If no route was found, return None.

    Parameters
    ----------
    request:
        Incoming Starlette/FastAPI request.

    Returns
    -------
    str:
        - If route template is available: returns template (best case)
          Example: "/recommend/recommend_movie_for_current_user"
        - If route is unknown, then None get's returned
    """
    # Starlette stores the matched route in request.scope["route"] once resolved.
    route = request.scope.get("route")
    if route is not None and hasattr(route, "path"):
        return str(route.path)

    return None


class PrometheusHTTPMetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware that records Prometheus HTTP metrics for every request. Only those calls, 
    where the routes are known will be tracked, all others will be ignored. That keeps
    the list of metrics clean.

    Implementation details
    ----------------------
    - We measure duration with time.perf_counter()
    - We increment/decrement the in-flight gauge using the SAME label set
      (method + path template) to avoid gauge leaks.
    - Because the route template is most reliably available AFTER call_next(),
      we compute the template then. This keeps labels stable and avoids "unknown".
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        This method get's automatically called whenever an HTTP request hits our API.
        We then receive the HTTP request and call_next http response.  

        Parameters
        ----------
        request:
            Incoming HTTP request.
        call_next:
            The HTTP response produced by your app.

        Returns
        -------
        Response:
            Response returned by downstream endpoint/middleware.
        """
        # Safe start time when the endpoint got called 
        start_time = time.perf_counter()

        # Get request method
        method = request.method

        # Increase the current number of requests counter, because request occurred 
        HTTP_REQUESTS_IN_FLIGHT.labels(method).inc()
        # Default status used if an exception occurred.
        status_code_str = "500"

        try:
            # Wait until the endpoint is finished and returned response
            response = await call_next(request)
            # Extract response status code and return it
            status_code_str = str(response.status_code)
            return response
        finally:
            # get the endpoint route corresponding to the request
            route_template = get_endpoint_route(request=request)

            # Decrease numb of current active endpoints, cause endpoint is finished
            HTTP_REQUESTS_IN_FLIGHT.labels(method).dec()

            # Check if route template exists, else return 
            if route_template is None or route_template in EXCLUDE_TEMPPLATES:
                return response
            
            # Measure duration time of endpoint call
            duration = time.perf_counter() - start_time

            # Save time measurement
            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=method,
                path=route_template,
            ).observe(duration)

            # Track total number of endpoint calls. 
            HTTP_REQUESTS_TOTAL.labels(
                method=method,
                path=route_template,
                status=status_code_str,
            ).inc()
            
