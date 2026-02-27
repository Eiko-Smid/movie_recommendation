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
        - If status_code == "404": "__not_found__"
        - Else: "__unmatched__"
    """
    # Starlette stores the matched route in request.scope["route"] once resolved.
    route = request.scope.get("route")
    if route is not None and hasattr(route, "path"):
        return str(route.path)

    return None


class PrometheusHTTPMetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware that records Prometheus HTTP metrics for every request.

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
            route_template = get_endpoint_route(
                request=request,
                status_code=status_code_str
            )

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
            
