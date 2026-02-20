"""
src/observability/metrics.py

This module contains Prometheus metrics definitions.
Keeping metrics in a dedicated module prevents circular imports
between FastAPI app startup (main.py) and routers (e.g. recommend.py).
"""

from prometheus_client import Counter


"""
Global Prometheus counter for recommendation requests.
Labels:
    endpoint: Name of endpoint
    result: success or failure
"""
RECOMMENDATION_REQUESTS = Counter(
    name="reco_requests_total",
    documentation="Total number of recommendation requests.",
    labelnames=["endpoint", "result"],
)

