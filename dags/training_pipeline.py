"""Scheduled training pipeline.

  extract_transform_load -> compute_features -> check_data_drift -> retraining_needed
      -> train_and_evaluate -> determine_production_model

1. Airflow pulls new openly licensed photos from iNaturalist into the database.
2. The model API computes extractor features for the new photos.
3. Evidently compares the photos the production model was trained on (historical) with the
   photos ingested since (recent). The report and metrics are logged to MLflow.
4. Training only continues if it is needed: drift, enough new data, no production model yet,
   or a forced run (Grafana alert, prediction drift, manual trigger with {"force": true}).
5. The API trains and evaluates; the run is logged to MLflow.
6. Airflow reads both scores from MLflow (new model and current production model, measured
   on the same test set) and moves the production tag if the new model is at least as good.
"""

import os
from datetime import datetime, timedelta

from airflow.decorators import dag, task

API_URL = os.getenv("API_URL", "http://api:8000")


@dag(
    dag_id="training_pipeline",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["species-id", "training"],
)
def training_pipeline():
    @task
    def extract_transform_load() -> dict:
        from species_id import config
        from species_id.ingest import ingest

        return ingest(new_per_taxon=config.NEW_IMAGES_PER_RUN)

    @task
    def compute_features() -> dict:
        import httpx

        response = httpx.post(f"{API_URL}/features", timeout=3600)
        response.raise_for_status()
        return response.json()

    @task
    def check_data_drift() -> dict:
        from species_id import drift

        return drift.check_training_data()

    @task.short_circuit
    def retraining_needed(drift_result: dict, dag_run=None) -> bool:
        from species_id import drift

        force = bool((dag_run.conf or {}).get("force"))
        decision = drift.retrain_decision(drift_result, force=force)
        print(f"Trigger reason: {(dag_run.conf or {}).get('reason', 'schedule')}")
        print(f"Decision: {decision}")
        return decision["retrain"]

    @task
    def train_and_evaluate() -> dict:
        import httpx

        response = httpx.post(f"{API_URL}/training", timeout=3600)
        response.raise_for_status()
        return response.json()

    @task
    def determine_production_model(run: dict) -> dict:
        from species_id import registry

        return registry.promote_if_better(run["run_id"], run["model_version"])

    drift_result = check_data_drift()
    extract_transform_load() >> compute_features() >> drift_result
    run = train_and_evaluate()
    retraining_needed(drift_result) >> run
    determine_production_model(run)


training_pipeline()
