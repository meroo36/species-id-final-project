"""Hourly check for drift in the photos submitted to the API.

  check_prediction_drift -> drift_detected -> trigger_training_pipeline

Evidently compares the features of photos the production model knows with the features
stored for recent predictions. Each prediction is checked once. The report and metrics are
logged to MLflow; if drift is detected, the training pipeline is started with force=true.
"""

from datetime import datetime

from airflow.decorators import dag, task
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


@dag(
    dag_id="prediction_drift",
    schedule="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["species-id", "monitoring"],
)
def prediction_drift():
    @task
    def check_prediction_drift() -> dict:
        from species_id import drift

        return drift.check_predictions()

    @task.short_circuit
    def drift_detected(result: dict) -> bool:
        print(f"Drift check: {result}")
        return bool(result.get("drift_detected"))

    trigger = TriggerDagRunOperator(
        task_id="trigger_training_pipeline",
        trigger_dag_id="training_pipeline",
        conf={"force": True, "reason": "prediction drift"},
    )

    drift_detected(check_prediction_drift()) >> trigger


prediction_drift()
