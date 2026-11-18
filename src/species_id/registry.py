"""MLflow Model Registry helpers: load the production model and promote new versions."""

import logging

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from species_id import config

log = logging.getLogger("registry")

META_ARTIFACT = "model_meta.json"


def client() -> MlflowClient:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return MlflowClient()


def production_version() -> str | None:
    """Version number currently tagged as production, or None if there is none yet."""
    try:
        return client().get_model_version_by_alias(config.MODEL_NAME, config.PRODUCTION_ALIAS).version
    except MlflowException:
        return None


def production_info() -> dict | None:
    """Describe the production model without loading it (version, extractor, params, metrics)."""
    version = production_version()
    if version is None:
        return None
    mlflow_client = client()
    mv = mlflow_client.get_model_version(config.MODEL_NAME, version)
    run = mlflow_client.get_run(mv.run_id)
    meta = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/{META_ARTIFACT}")
    return {
        "version": str(version),
        "run_id": mv.run_id,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        **meta,
    }


def versions() -> list[dict]:
    """Every registered model version with its score and promotion status, newest first."""
    mlflow_client = client()
    production = production_version()
    out = []
    for mv in mlflow_client.search_model_versions(f"name='{config.MODEL_NAME}'"):
        run = mlflow_client.get_run(mv.run_id).data
        out.append(
            {
                "version": int(mv.version),
                "status": mv.tags.get("validation_status", "pending"),
                "production": str(mv.version) == production,
                "macro_f1": run.metrics.get("macro_f1"),
                "accuracy": run.metrics.get("accuracy"),
                "production_macro_f1": run.metrics.get("production_macro_f1"),
                "n_train": int(run.params.get("n_train", 0)),
                "dataset_md5": run.params.get("dataset_md5"),
            }
        )
    return sorted(out, key=lambda v: v["version"], reverse=True)


def load_model(version: str) -> dict:
    """Load a registered model version with the metadata needed to serve it."""
    import mlflow.sklearn  # imported here so light clients (Airflow) do not need scikit-learn

    mv = client().get_model_version(config.MODEL_NAME, version)
    meta = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/{META_ARTIFACT}")
    head = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}/{version}")
    return {"head": head, "version": str(version), **meta}


def promote_if_better(run_id: str, version: str) -> dict:
    """Give the production tag to `version` unless it is worse than the production model.

    Both scores come from the training run, where the production model was re-evaluated on
    the same test set as the new one. A new model within PROMOTION_TOLERANCE of production is
    still promoted: the difference is noise on a test set this size, and the new model has
    seen more recent data.
    """
    mlflow_client = client()
    metrics = mlflow_client.get_run(run_id).data.metrics
    new_score = metrics[config.PROMOTION_METRIC]
    production_score = metrics.get(f"production_{config.PROMOTION_METRIC}")
    promoted = (
        production_score is None
        or new_score >= production_score - config.PROMOTION_TOLERANCE
    )

    if promoted:
        mlflow_client.set_registered_model_alias(config.MODEL_NAME, config.PRODUCTION_ALIAS, version)
    mlflow_client.set_model_version_tag(
        config.MODEL_NAME, version, "validation_status", "promoted" if promoted else "rejected"
    )
    decision = {
        "version": str(version),
        "promoted": promoted,
        "new_score": new_score,
        "production_score": production_score,
        "metric": config.PROMOTION_METRIC,
        "tolerance": config.PROMOTION_TOLERANCE,
    }
    log.info("Promotion decision: %s", decision)
    return decision
