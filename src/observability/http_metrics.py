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
from typing import Optional

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
    ["method", "path"],
)


# =============================================================================
# Helper functions
# =============================================================================

def get_path_label(request: Request, status_code: str) -> str:
    """
    Determine a low-cardinality path label for Prometheus metrics.

    Parameters
    ----------
    request:
        Incoming Starlette/FastAPI request.
    status_code:
        HTTP status code as string (e.g. "200", "404", "500").

    Returns
    -------
    str:
        - If route template is available: returns template (best case)
          Example: "/recommend/recommend_movie_for_current_user"
        - If status_code == "404": "__not_found__"
        - Else: "__unmatched__"

    Notes
    -----
    We intentionally do NOT use request.url.path or request.scope["path"]
    because those can contain IDs and lead to cardinality explosion.
    """
    # Starlette stores the matched route in request.scope["route"] once resolved.
    route = request.scope.get("route")
    if route is not None and hasattr(route, "path"):
        return str(route.path)

    # If no route is available, at least distinguish 404 from other issues.
    if status_code == "404":
        return "__not_found__"

    return "__unmatched__"


# =============================================================================
# Middleware class
# =============================================================================

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
        Process a request, record metrics, then return the response.

        Parameters
        ----------
        request:
            Incoming request.
        call_next:
            Callable that forwards the request to the downstream app.

        Returns
        -------
        Response:
            Response returned by downstream endpoint/middleware.
        """
        method = request.method

        # Default status used if an exception bubbles up.
        status_code_str: str = "500"

        # We initially don't know the final route template.
        # We'll determine it after call_next() (or in finally).
        path_label: str = "__unmatched__"

        # Track start time for latency.
        start = time.perf_counter()

        # We cannot reliably label in-flight with the final route template
        # *before* routing happens. However, once routing is done (after call_next),
        # we can use the template. To ensure inc/dec labels match, we:
        # - resolve the route template after call_next
        # - then inc and dec immediately in finally (net 0) wouldn't help
        #
        # So for correct "in-flight" we need the template early.
        # FastAPI often resolves route before this middleware runs, but not always.
        #
        # We attempt to resolve early; if missing we use "__unmatched__".
        early_route = request.scope.get("route")
        if early_route is not None and hasattr(early_route, "path"):
            path_label = str(early_route.path)

        # Increment in-flight with whatever best label we have right now.
        HTTP_REQUESTS_IN_FLIGHT.labels(method=method, path=path_label).inc()

        try:
            # Execute downstream app (routing + endpoint)
            response = await call_next(request)
            status_code_str = str(response.status_code)
            return response

        finally:
            # Prefer the final, best label (template if available).
            final_path_label = get_path_label(request, status_code_str)

            # Measure duration
            duration = time.perf_counter() - start

            # Observe duration and count requests using FINAL label
            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=method,
                path=final_path_label,
            ).observe(duration)

            HTTP_REQUESTS_TOTAL.labels(
                method=method,
                path=final_path_label,
                status=status_code_str,
            ).inc()

            # Decrement in-flight using the SAME label we incremented with.
            HTTP_REQUESTS_IN_FLIGHT.labels(method=method, path=path_label).dec()


            