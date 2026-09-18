"""Model-independent tracking contracts and backend adapters."""

from .backends import CoTrackerBackend
from .core import (
    BaseTrackingPredictionBackend,
    COTRACKER_BACKEND_ID,
    TRACKING_BACKEND_CONTRACT_VERSION,
    TrackingBackend,
    TrackingPredictionBackend,
    TrackingOutput,
    infer_tracking_grid_size,
    normalize_tracking_backend_identity,
    prepare_tracking_query_points,
    resolve_tracking_query_mode,
    run_tracking_backend,
    run_tracking_prediction_backend,
    tracking_backend_identity,
    validate_external_tracking_runtime_identity,
)

__all__ = [
    "BaseTrackingPredictionBackend",
    "COTRACKER_BACKEND_ID",
    "TRACKING_BACKEND_CONTRACT_VERSION",
    "CoTrackerBackend",
    "TrackingBackend",
    "TrackingPredictionBackend",
    "TrackingOutput",
    "infer_tracking_grid_size",
    "normalize_tracking_backend_identity",
    "prepare_tracking_query_points",
    "resolve_tracking_query_mode",
    "run_tracking_backend",
    "run_tracking_prediction_backend",
    "tracking_backend_identity",
    "validate_external_tracking_runtime_identity",
]
