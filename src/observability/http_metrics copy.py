from __future__ import annotations
from typing import Callable, Optional

import time

from fastapi import Request, Response
from prometheus_client import Counter, Histogram, Gauge

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

#____________________________________________________________________________________
# Prometheus metric definitions (GLOBAL, created once per process)
#____________________________________________________________________________________

"""
Counts total HTTP requests.

Labels:
    method: HTTP method (GET, POST, ...)
    path: FastAPI route template (e.g. "/recommend/recommend_movie_by_id")
    status: HTTP status code as string (e.g. "200", "401", "500")
"""
HTTP_REQUEST_TOTAL = Counter(
    name="http_request_total",
    documentation="Total number of HTTP requests.",
    labelnames=["method", "path", "status"],
)


"""
Measures HTTP request duration (latency) in seconds.

Labels:
    method: HTTP method
    path: FastAPI route template

Notes:
- Histogram gives you latency distribution and percentiles (p95 etc.) in Grafana.
"""
HTTP_REQUEST_DURATION = Histogram(
    name="http_request_duration_sec",
    documentation="HTTP request duration in seconds.",
    labelnames=["method", "path"],
)


"""
Tracks how many HTTP requests are currently being processed.

Labels:
    method: HTTP method
    path: FastAPI route template
"""
HTTP_REQUEST_IN_FLIGHT = Gauge(
    name="http_request_in_flight",
    documentation="Number of HTTP requests on flight.",
    labelnames=["method", "path"],
)


#____________________________________________________________________________________
# Helper: find a stable route template for labeling
#____________________________________________________________________________________

# def _get_route_template(request: Request) -> str:
#     # get route of http endpoint
#     route = request.scope.get("route")

#     # Return route as string if exists and is indeed route obj
#     if route is not None and hasattr(route, "path"):
#         return str(route.path)
    
#     route_template = str(request.url.path)
#     if route_template is not None:
#         return route_template

#     return "unknown"

def _get_route_template(request: Request) -> str:
    """
    Get the matched route template (low cardinality), e.g. "/health" or
    "/admin/users/{user_id}/role".

    Falls back to "unknown" only if route resolution is unavailable.
    """
    route = request.scope.get("route")
    if route is not None and hasattr(route, "path"):
        return str(route.path)
    return "unknown"


def get_path_label(request: Request, status_code: str) -> str:
    """
    Determine the Prometheus 'path' label.

    Parameters
    ----------
    request:
        Incoming request object.
    status_code:
        HTTP status code as string (e.g. "200", "404").

    Returns
    -------
    str:
        A low-cardinality label value:
        - route template (preferred), e.g. "/admin/users/{user_id}/role"
        - "__not_found__" for 404s
        - "__unmatched__" when route isn't available
    """
    route = request.scope.get("route")
    if route is not None and hasattr(route, "path"):
        return str(route.path)

    if status_code == "404":
        return "__not_found__"

    return "__unmatched__"


class PrometheusHTTPMetricsMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware that records Prometheus HTTP metrics for every request.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Dispatches the request, measures duration, and updates metrics.

        Parameters
        ----------
        request:
            Incoming HTTP request.
        call_next:
            Callable that executes the downstream app and returns a Response.

        Returns
        -------
        Response:
            The HTTP response produced by your app.
        """
        method = request.method

        # We will resolve the route template AFTER call_next (more reliable)
        # but we still want in-flight tracking: use raw path *temporarily*,
        # then also record final metrics with template.
        start = time.perf_counter()

        response: Response | None = None
        status_code = "500"
        path_template = "unknown"

        try:
            response = await call_next(request)
            status_code = str(response.status_code)
            return response
        finally:
            # Now route should be resolved
            path_template = _get_route_template(request)
            duration = time.perf_counter() - start

            HTTP_REQUEST_DURATION.labels(method=method, path=path_template).observe(duration)
            HTTP_REQUEST_TOTAL.labels(method=method, path=path_template, status=status_code).inc()


# async def prometheus_http_middleware(request: Request, call_next: Callable) -> Response:
#     method = request.method

#     # Get current route
#     path_template = _get_route_template(request=request)

#     # Add one to current number of 
#     HTTP_REQUEST_IN_FLIGHT.labels(method=method, path=path_template).inc()

#     # Get start time infractional seconds
#     start = time.perf_counter()
#     status_code_str: Optional[str] = None

#     try:
#         response = await call_next(request)
#         status_code_str = str(response.status_code)
#         return response
#     except Exception:
#         status_code_str = "500"
#         raise
#     finally:
#         # Compute duration time of endpoint and save in histogram
#         duration = time.perf_counter() - start
#         HTTP_REQUEST_DURATION.labels(method=method, path=path_template).observe(duration)
        
#         if status_code_str is None:
#             status_code_str = "500"
        
#         # Increment total number of request for endpoint
#         HTTP_REQUEST_TOTAL.labels(method=method, path=path_template, status=status_code_str).inc()

#         # Decrement in flight at end if endpoint call
#         HTTP_REQUEST_IN_FLIGHT.labels(method=method, path=path_template).dec()