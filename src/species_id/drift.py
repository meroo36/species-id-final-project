"""Data drift detection with Evidently.

Images are compared through their extractor features. The 1280-dimensional vectors are
projected onto the main PCA axes of the reference sample, and Evidently tests each axis for a
distribution shift (Kolmogorov-Smirnov, p < 0.05). Drift is declared when at least half of the axes have drifted.

Two checks use this:
- training data: photos the production model was trained on vs photos ingested since;
- predictions: photos known to the model vs photos recently submitted to the API.

Every check that runs is logged to MLflow (metrics + HTML report) and to `drift_checks`.
"""

import logging
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from species_id import config, db, registry

log = logging.getLogger("drift")

EXPERIMENT = "species-id-monitoring"
N_COMPONENTS = 10      # PCA axes compared
MIN_ROWS = 30          # below this, a sample is too small to test
MAX_REFERENCE = 2000   # reference rows sampled per check


def _skipped(reason: str) -> dict:
    log.info("Drift check skipped: %s", reason)
    return {"status": "skipped", "reason": reason, "drift_detected": False}


def _vectors(rows) -> np.ndarray:
    return np.array([r[0] for r in rows], dtype=np.float32)


def run_report(reference: np.ndarray, current: np.ndarray) -> tuple[dict, object]:
    """Compare two feature samples. Returns Evidently's dataset-level result and the report."""
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report
    from sklearn.decomposition import PCA

    pca = PCA(n_components=N_COMPONENTS, random_state=0).fit(reference)
    columns = [f"pc_{i + 1}" for i in range(N_COMPONENTS)]
    # Kolmogorov-Smirnov accounts for sample size; Evidently's default for large references
    # (Wasserstein distance) raises false alarms when the current sample is small.
    report = Report(metrics=[DataDriftPreset(num_stattest="ks", num_stattest_threshold=0.05)])
    report.run(
        reference_data=pd.DataFrame(pca.transform(reference), columns=columns),
        current_data=pd.DataFrame(pca.transform(current), columns=columns),
    )
    return report.as_dict()["metrics"][0]["result"], report


def _record(kind: str, reference: np.ndarray, current: np.ndarray, window_end, model: dict) -> dict:
    """Run the report, log it to MLflow and the database, and return a summary."""
    result, report = run_report(reference, current)
    summary = {
        "status": "checked",
        "kind": kind,
        "n_reference": len(reference),
        "n_current": len(current),
        "drift_share": float(result["share_of_drifted_columns"]),
        "n_drifted_features": int(result["number_of_drifted_columns"]),
        "drift_detected": bool(result["dataset_drift"]),
        "production_version": model["version"],
    }

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"{kind}-drift") as run, tempfile.TemporaryDirectory() as tmp:
        mlflow.set_tags({"kind": kind, "production_version": model["version"]})
        mlflow.log_metrics(
            {
                "drift_share": summary["drift_share"],
                "n_drifted_features": summary["n_drifted_features"],
                "drift_detected": int(summary["drift_detected"]),
                "n_reference": summary["n_reference"],
                "n_current": summary["n_current"],
            }
        )
        html = Path(tmp) / f"{kind}_drift_report.html"
        report.save_html(str(html))
        mlflow.log_artifact(str(html))
    summary["mlflow_run_id"] = run.info.run_id

    with db.connect() as conn:
        conn.execute(
            """INSERT INTO drift_checks (kind, window_end, n_reference, n_current, drift_share,
                                         drift_detected, mlflow_run_id)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (kind, window_end, summary["n_reference"], summary["n_current"],
             summary["drift_share"], summary["drift_detected"], summary["mlflow_run_id"]),
        )
    log.info("Drift check: %s", summary)
    return summary


def check_training_data() -> dict:
    """Historical photos (seen by the production model) vs photos ingested since."""
    db.init_schema()
    model = registry.production_info()
    if model is None:
        return _skipped("no production model yet")
    cutoff = model["params"].get("data_cutoff")
    if cutoff is None:
        return _skipped("production model has no recorded data cutoff")

    with db.connect() as conn:
        reference = conn.execute(
            """SELECT e.vector FROM embeddings e JOIN photos p USING (photo_id)
               WHERE e.extractor = %s AND p.ingested_at <= %s::timestamptz
               ORDER BY random() LIMIT %s""",
            (model["extractor"], cutoff, MAX_REFERENCE),
        ).fetchall()
        current = conn.execute(
            """SELECT e.vector, p.ingested_at FROM embeddings e JOIN photos p USING (photo_id)
               WHERE e.extractor = %s AND p.ingested_at > %s::timestamptz""",
            (model["extractor"], cutoff),
        ).fetchall()
    if len(current) < MIN_ROWS or len(reference) < MIN_ROWS:
        return _skipped(f"not enough data ({len(reference)} historical, {len(current)} recent)")
    window_end = max(r[1] for r in current)
    return _record("training", _vectors(reference), _vectors(current), window_end, model)


def check_predictions() -> dict:
    """Photos known to the production model vs photos submitted to the API since the last check."""
    db.init_schema()
    model = registry.production_info()
    if model is None:
        return _skipped("no production model yet")

    with db.connect() as conn:
        current = conn.execute(
            """SELECT vector, created_at FROM predictions
               WHERE extractor = %s AND created_at > COALESCE(
                   (SELECT max(window_end) FROM drift_checks WHERE kind = 'prediction'),
                   '-infinity')""",
            (model["extractor"],),
        ).fetchall()
        if len(current) < MIN_ROWS:
            return _skipped(f"only {len(current)} new predictions (need {MIN_ROWS})")
        reference = conn.execute(
            """SELECT e.vector FROM embeddings e WHERE e.extractor = %s
               ORDER BY random() LIMIT %s""",
            (model["extractor"], MAX_REFERENCE),
        ).fetchall()
    if len(reference) < MIN_ROWS:
        return _skipped("not enough reference data")
    window_end = max(r[1] for r in current)
    return _record("prediction", _vectors(reference), _vectors(current), window_end, model)


def retrain_decision(drift: dict, force: bool = False) -> dict:
    """Decide whether the training pipeline should train a new model, and say why."""
    reasons = []
    model = registry.production_info()
    if force:
        reasons.append("forced by trigger")
    if model is None:
        reasons.append("no production model yet")
    elif model["params"].get("data_cutoff") is None:
        reasons.append("production model has no recorded data cutoff")
    else:
        with db.connect() as conn:
            new_photos = conn.execute(
                "SELECT count(*) FROM photos WHERE ingested_at > %s::timestamptz",
                (model["params"]["data_cutoff"],),
            ).fetchone()[0]
        if new_photos >= config.RETRAIN_MIN_NEW_PHOTOS:
            reasons.append(f"{new_photos} new photos since the production model was trained")
    if drift.get("drift_detected"):
        reasons.append(f"data drift detected (share {drift['drift_share']:.2f})")
    decision = {"retrain": bool(reasons), "reasons": reasons}
    log.info("Retrain decision: %s", decision)
    return decision
