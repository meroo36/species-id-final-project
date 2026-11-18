"""Database access helpers."""

from importlib import resources

import psycopg

from species_id import config


def connect() -> psycopg.Connection:
    return psycopg.connect(config.POSTGRES_DSN)


def init_schema() -> None:
    """Create tables if they do not exist."""
    sql = resources.files("species_id").joinpath("schema.sql").read_text()
    with connect() as conn:
        conn.execute(sql)
