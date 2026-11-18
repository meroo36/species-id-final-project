FROM python:3.12-slim

# SQLAlchemy 2.1 removed a class MLflow 2.x imports, so it is capped here.
RUN pip install --no-cache-dir "mlflow>=2.20,<3" "sqlalchemy<2.1" psycopg2-binary

EXPOSE 5000
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri \"$BACKEND_STORE_URI\" --artifacts-destination /mlartifacts"]
