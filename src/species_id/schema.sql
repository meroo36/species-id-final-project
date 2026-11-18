-- Species ID database schema. Idempotent: safe to run more than once.

CREATE TABLE IF NOT EXISTS taxa (
    taxon_id        INTEGER PRIMARY KEY,          -- iNaturalist taxon id
    scientific_name TEXT NOT NULL UNIQUE,
    common_name     TEXT
);

-- One row per photo. License and attribution are kept so every image stays traceable.
CREATE TABLE IF NOT EXISTS photos (
    photo_id       BIGINT PRIMARY KEY,            -- iNaturalist photo id
    observation_id BIGINT NOT NULL,
    taxon_id       INTEGER NOT NULL REFERENCES taxa (taxon_id),
    url            TEXT NOT NULL,
    license        TEXT NOT NULL,
    attribution    TEXT NOT NULL,
    observed_on    DATE,
    local_path     TEXT NOT NULL,
    split          TEXT NOT NULL CHECK (split IN ('train', 'test')),
    ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS photos_taxon_idx ON photos (taxon_id);

-- Frozen-extractor features, cached so retraining only fits the classification head.
CREATE TABLE IF NOT EXISTS embeddings (
    photo_id  BIGINT NOT NULL REFERENCES photos (photo_id) ON DELETE CASCADE,
    extractor TEXT NOT NULL,
    vector    REAL[] NOT NULL,
    PRIMARY KEY (photo_id, extractor)
);

-- Every prediction served by the API, with its features, for monitoring and drift checks.
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id   BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version   TEXT NOT NULL,
    extractor       TEXT NOT NULL,
    predicted_taxon INTEGER NOT NULL REFERENCES taxa (taxon_id),
    confidence      REAL NOT NULL,
    needs_review    BOOLEAN NOT NULL,
    vector          REAL[] NOT NULL
);

-- One row per drift check that had enough data to run (see drift.py).
CREATE TABLE IF NOT EXISTS drift_checks (
    check_id       BIGSERIAL PRIMARY KEY,
    kind           TEXT NOT NULL CHECK (kind IN ('training', 'prediction')),
    checked_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    window_end     TIMESTAMPTZ NOT NULL,           -- newest row included in the current sample
    n_reference    INTEGER NOT NULL,
    n_current      INTEGER NOT NULL,
    drift_share    REAL NOT NULL,                  -- share of drifted features
    drift_detected BOOLEAN NOT NULL,
    mlflow_run_id  TEXT
);
