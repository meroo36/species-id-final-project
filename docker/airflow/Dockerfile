FROM apache/airflow:2.10.5-python3.12

# Light dependencies only: the DAGs run ingestion and drift checks themselves, call the API
# for features and training, and talk to MLflow for the promotion decision. The heavy ML
# stack (PyTorch) stays in the API image.
RUN pip install --no-cache-dir \
    "psycopg[binary]>=3.2" \
    "httpx>=0.27" \
    "python-dotenv>=1.0" \
    "mlflow-skinny>=2.20,<3" \
    "evidently>=0.6,<0.7"
