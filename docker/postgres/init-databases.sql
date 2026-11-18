-- Runs once, when the Postgres volume is first created.
-- The application database (POSTGRES_DB) is created by the image itself.
CREATE DATABASE mlflow;
CREATE DATABASE airflow;
