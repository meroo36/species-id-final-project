"""Identify the species in a photograph with the production model from the MLflow registry.

Run with:  uv run python -m species_id.predict path/to/photo.jpg
"""

import json
import sys
import time

from PIL import Image

from species_id import config, db, metrics, registry
from species_id.features import embed

REFRESH_S = 30  # how often to check whether the production tag moved to a new version
_cache: dict = {"checked_at": 0.0, "model": None}


def load_model() -> dict:
    """Return the production model, picking up a newly promoted version without a restart."""
    if time.monotonic() - _cache["checked_at"] > REFRESH_S or _cache["model"] is None:
        version = registry.production_version()
        if version is None:
            raise LookupError("No production model yet. Run the training pipeline first.")
        if _cache["model"] is None or _cache["model"]["version"] != version:
            _cache["model"] = registry.load_model(version)
        _cache["checked_at"] = time.monotonic()
    return _cache["model"]


def predict(image: Image.Image, top_k: int = 3, store: bool = True) -> dict:
    """Return the top species with confidence, flagging uncertain results for human review."""
    model = load_model()
    vector = embed([image], model["extractor"])
    proba = model["head"].predict_proba(vector)[0]
    ranked = proba.argsort()[::-1][:top_k]
    confidence = float(proba[ranked[0]])
    best = model["classes"][ranked[0]]
    needs_review = confidence < config.CONFIDENCE_THRESHOLD

    metrics.PREDICTIONS.labels(best["scientific_name"], str(needs_review).lower()).inc()
    metrics.CONFIDENCE.observe(confidence)
    metrics.MODEL_VERSION.set(int(model["version"]))

    if store:
        with db.connect() as conn:
            conn.execute(
                """INSERT INTO predictions (model_version, extractor, predicted_taxon,
                                            confidence, needs_review, vector)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (model["version"], model["extractor"], best["taxon_id"], confidence,
                 needs_review, vector[0].tolist()),
            )

    return {
        "model_version": model["version"],
        "prediction": best,
        "confidence": confidence,
        "needs_review": needs_review,
        "top": [{**model["classes"][i], "confidence": float(proba[i])} for i in ranked],
    }


if __name__ == "__main__":
    print(json.dumps(predict(Image.open(sys.argv[1]), store=False), indent=2))
