"""Inference API.

Run with:  uv run uvicorn species_id.api:app --port 8000
"""

import io
import logging
import threading
from typing import Annotated

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from prometheus_fastapi_instrumentator import Instrumentator

from species_id import config, db, metrics, registry
from species_id import predict as predict_module
from species_id import training as training_module

log = logging.getLogger("api")

app = FastAPI(title="Species ID", version="0.3.0")
Instrumentator(excluded_handlers=["/metrics", "/health"]).instrument(app).expose(
    app, include_in_schema=False
)
_training_lock = threading.Lock()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model")
def model() -> dict:
    """Describe the production model: version, extractor, species, metrics."""
    info = registry.production_info()
    if info is None:
        raise HTTPException(503, "No production model yet. Run the training pipeline first.")
    return info


@app.get("/stats")
def stats() -> dict:
    """Live figures about the data, the predictions, the drift checks and the model versions."""
    with db.connect() as conn:
        species = conn.execute(
            """SELECT t.scientific_name, t.common_name, count(*),
                      count(*) FILTER (WHERE p.split = 'test')
               FROM photos p JOIN taxa t USING (taxon_id) GROUP BY 1, 2 ORDER BY 1"""
        ).fetchall()
        licenses = conn.execute(
            "SELECT license, count(*) FROM photos GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()
        n_predictions, n_review = conn.execute(
            "SELECT count(*), count(*) FILTER (WHERE needs_review) FROM predictions"
        ).fetchone()
        checks = conn.execute(
            """SELECT checked_at, kind, n_current, drift_share, drift_detected
               FROM drift_checks ORDER BY checked_at DESC LIMIT 10"""
        ).fetchall()
    return {
        "species": [
            {"scientific_name": s, "common_name": c, "photos": n, "test_photos": t}
            for s, c, n, t in species
        ],
        "licenses": [{"license": name, "photos": n} for name, n in licenses],
        "predictions": {"total": n_predictions, "needs_review": n_review},
        "drift_checks": [
            {"checked_at": at.isoformat(timespec="seconds"), "kind": kind, "n_current": n,
             "drift_share": share, "drift_detected": detected}
            for at, kind, n, share, detected in checks
        ],
        "versions": registry.versions(),
    }


@app.post("/features")
def features() -> dict:
    """Compute extractor features for photos that do not have them yet."""
    extractor = (registry.production_info() or {}).get("extractor", config.EXTRACTOR)
    return {
        "extractor": extractor,
        "computed": training_module.compute_missing_embeddings(extractor),
    }


@app.post("/training")
def training(promote: bool = False) -> dict:
    """Train and evaluate a new model on the data currently in the database.

    The run is tracked in MLflow and registered as a new model version. With `promote=true`
    the new version also takes the production tag if it beats the current production model;
    in the scheduled pipeline that decision is made by Airflow instead.
    """
    if not _training_lock.acquire(blocking=False):
        raise HTTPException(409, "A training run is already in progress.")
    try:
        metrics.TRAININGS.inc()
        return training_module.train(promote=promote)
    except RuntimeError as error:
        raise HTTPException(400, str(error)) from error
    finally:
        _training_lock.release()


@app.post("/predict")
def predict(file: Annotated[UploadFile, File()]) -> dict:
    """Identify the species in an uploaded photograph."""
    try:
        image = Image.open(io.BytesIO(file.file.read()))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(400, "The uploaded file is not a readable image.") from error
    try:
        return predict_module.predict(image)
    except LookupError as error:
        raise HTTPException(503, str(error)) from error


@app.post("/webhooks/grafana")
def grafana_webhook(payload: dict) -> dict:
    """Receive Grafana alert notifications.

    Firing alerts labelled `action=retrain` start the Airflow training pipeline; everything
    else is only logged.
    """
    firing = [a for a in payload.get("alerts", []) if a.get("status") == "firing"]
    retrain = [a for a in firing if a.get("labels", {}).get("action") == "retrain"]
    log.warning("Grafana webhook: %d firing alert(s), %d asking for retraining",
                len(firing), len(retrain))
    if not retrain:
        return {"triggered": False, "firing": len(firing)}

    alert = retrain[0]["labels"].get("alertname", "unknown")
    response = httpx.post(
        f"{config.AIRFLOW_URL}/api/v1/dags/{config.TRAINING_DAG_ID}/dagRuns",
        json={"conf": {"force": True, "reason": f"grafana alert: {alert}"}},
        auth=(config.AIRFLOW_ADMIN_USER, config.AIRFLOW_ADMIN_PASSWORD),
        timeout=30,
    )
    if response.status_code >= 400:
        raise HTTPException(502, f"Airflow refused the trigger: {response.text}")
    metrics.RETRAIN_TRIGGERS.labels(alert).inc()
    return {"triggered": True, "alert": alert, "dag_run_id": response.json()["dag_run_id"]}
