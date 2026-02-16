from __future__ import annotations

import time
from contextlib import contextmanager

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response
from prometheus_client import Gauge

REQUESTS = Counter(
    "creditlimit_requests_total",
    "Total requests",
    ["endpoint", "status"],
)

LATENCY = Histogram(
    "creditlimit_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

MODEL_LOAD_SECONDS = Gauge(
    "creditlimit_model_load_seconds",
    "Time taken to load models at startup",
)

PREDICTIONS = Counter(
    "creditlimit_predictions_total",
    "Prediction counts",
    ["action"],
)



INFERENCE_STAGE_LATENCY = Histogram(
    "creditlimit_inference_stage_latency_seconds",
    "Latency per model stage",
    ["stage"],  # gate | dir | magnitude | explain
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

ACTIVE_BUNDLE = Gauge(
    "creditlimit_active_model_bundle",
    "Active model bundle version/sha (set to 1 for active label)",
    ["bundle_id"],  # e.g. ckpt hash, timestamp, “champion”
)


def record_request(endpoint: str, status_code: int) -> None:
    REQUESTS.labels(endpoint=endpoint, status=str(status_code)).inc()


def record_prediction(action: str) -> None:
    PREDICTIONS.labels(action=action).inc()


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@contextmanager
def track_latency(endpoint: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)
