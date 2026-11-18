"""Settings read from environment variables (see .env.example)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

POSTGRES_DSN = (
    f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
    f"port={os.getenv('POSTGRES_PORT', '5432')} "
    f"dbname={os.getenv('POSTGRES_DB', 'species_id')} "
    f"user={os.getenv('POSTGRES_USER', 'species')} "
    f"password={os.getenv('POSTGRES_PASSWORD', 'species')}"
)

# Images live under DATA_DIR/images; the database stores paths relative to DATA_DIR so the
# same rows work on the host and inside containers.
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
IMAGE_DIR = DATA_DIR / "images"

EXTRACTOR = os.getenv("EXTRACTOR", "mobilenetv3_large_100.ra_in1k")

IMAGES_PER_TAXON = int(os.getenv("IMAGES_PER_TAXON", "300"))
NEW_IMAGES_PER_RUN = int(os.getenv("NEW_IMAGES_PER_RUN", "20"))
ALLOWED_LICENSES = os.getenv("ALLOWED_LICENSES", "cc0,cc-by,cc-by-nc").split(",")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))

# Retrain when this many photos have arrived since the production model was trained,
# even without drift.
RETRAIN_MIN_NEW_PHOTOS = int(os.getenv("RETRAIN_MIN_NEW_PHOTOS", "1000"))

# Used by the API to start the training pipeline (Grafana alert webhook).
AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://localhost:8080")
AIRFLOW_ADMIN_USER = os.getenv("AIRFLOW_ADMIN_USER", "admin")
AIRFLOW_ADMIN_PASSWORD = os.getenv("AIRFLOW_ADMIN_PASSWORD", "admin")
TRAINING_DAG_ID = "training_pipeline"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "species-id")
MODEL_NAME = os.getenv("MODEL_NAME", "species-id")
PRODUCTION_ALIAS = "production"
PROMOTION_METRIC = "macro_f1"
# A new model this close to production (or better) is promoted; see registry.promote_if_better.
PROMOTION_TOLERANCE = float(os.getenv("PROMOTION_TOLERANCE", "0.005"))

# Level 1 perimeter: ten well-represented, visually distinct species.
# Ids are resolved from these names against the iNaturalist API at ingestion time.
TAXA = [
    "Apis mellifera",          # Western honey bee
    "Danaus plexippus",        # Monarch butterfly
    "Harmonia axyridis",       # Asian lady beetle
    "Anas platyrhynchos",      # Mallard
    "Turdus migratorius",      # American robin
    "Ardea herodias",          # Great blue heron
    "Sciurus carolinensis",    # Eastern gray squirrel
    "Odocoileus virginianus",  # White-tailed deer
    "Taraxacum officinale",    # Common dandelion
    "Amanita muscaria",        # Fly agaric
]
