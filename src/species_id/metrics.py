"""Prometheus metrics specific to the model (HTTP metrics come from the instrumentator)."""

from prometheus_client import Counter, Gauge, Histogram

PREDICTIONS = Counter(
    "species_predictions_total",
    "Predictions served, by predicted species and human-review flag.",
    ["species", "needs_review"],
)
CONFIDENCE = Histogram(
    "species_prediction_confidence",
    "Confidence of the top prediction.",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)
MODEL_VERSION = Gauge(
    "species_model_version", "Version of the production model currently served."
)
TRAININGS = Counter("species_trainings_total", "Training runs executed by the API.")
RETRAIN_TRIGGERS = Counter(
    "species_retrain_triggers_total", "Training pipeline runs started by an alert.", ["alert"]
)
