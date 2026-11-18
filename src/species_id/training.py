"""Train the classification head on frozen-extractor features read from the database.

Every run is tracked in MLflow (parameters, metrics, dataset hash, model) and registered as a
new model version. The current production model is re-evaluated on the same test set so the
two can be compared fairly.

Run with:  uv run python -m species_id.training
"""

import json
import logging

import mlflow
import mlflow.sklearn
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from species_id import config, db, registry, versioning
from species_id.features import embed

log = logging.getLogger("training")

EMBED_CHUNK = 256
HEAD_C = 0.1


def compute_missing_embeddings(extractor: str) -> int:
    """Embed every photo that has no cached features for this extractor yet."""
    with db.connect() as conn:
        missing = conn.execute(
            """SELECT p.photo_id, p.local_path FROM photos p
               LEFT JOIN embeddings e ON e.photo_id = p.photo_id AND e.extractor = %s
               WHERE e.photo_id IS NULL ORDER BY p.photo_id""",
            (extractor,),
        ).fetchall()
        for start in range(0, len(missing), EMBED_CHUNK):
            chunk = missing[start : start + EMBED_CHUNK]
            vectors = embed((Image.open(config.DATA_DIR / path) for _, path in chunk), extractor)
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO embeddings (photo_id, extractor, vector) VALUES (%s, %s, %s)",
                    [
                        (pid, extractor, vec.tolist())
                        for (pid, _), vec in zip(chunk, vectors, strict=True)
                    ],
                )
            conn.commit()
            log.info("Embedded %d/%d photos", start + len(chunk), len(missing))
    return len(missing)


def load_dataset(extractor: str):
    compute_missing_embeddings(extractor)
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT e.vector, p.taxon_id, p.split FROM embeddings e
               JOIN photos p USING (photo_id) WHERE e.extractor = %s""",
            (extractor,),
        ).fetchall()
        taxa = conn.execute("SELECT taxon_id, scientific_name, common_name FROM taxa").fetchall()
    X = np.array([r[0] for r in rows], dtype=np.float32)
    y = np.array([r[1] for r in rows])
    is_test = np.array([r[2] == "test" for r in rows])
    names = {t[0]: {"scientific_name": t[1], "common_name": t[2]} for t in taxa}
    return X, y, is_test, names


def evaluate(head, X: np.ndarray, y: np.ndarray) -> dict:
    proba = head.predict_proba(X)
    predicted = head.classes_[proba.argmax(axis=1)]
    confident = proba.max(axis=1) >= config.CONFIDENCE_THRESHOLD
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
        # Share of test photos sent to human review, and accuracy on the ones that are not.
        "review_rate": float(1 - confident.mean()),
        "accuracy_when_confident": float(
            accuracy_score(y[confident], predicted[confident]) if confident.any() else 0.0
        ),
    }


def evaluate_production(extractor: str, X_test: np.ndarray, y_test: np.ndarray) -> dict | None:
    """Score the current production model on the current test set (None if there is none)."""
    version = registry.production_version()
    if version is None:
        return None
    production = registry.load_model(version)
    if production["extractor"] != extractor:
        X, y, is_test, _ = load_dataset(production["extractor"])
        X_test, y_test = X[is_test], y[is_test]
    return {"version": version, **evaluate(production["head"], X_test, y_test)}


def train(extractor: str = config.EXTRACTOR, promote: bool = False) -> dict:
    """Train, evaluate, track and register a model. Returns the run summary."""
    db.init_schema()
    dataset = versioning.snapshot_dataset()
    X, y, is_test, names = load_dataset(extractor)
    if len(X) == 0:
        raise RuntimeError("No data in the database. Run the ingestion script first.")

    head = make_pipeline(StandardScaler(), LogisticRegression(C=HEAD_C, max_iter=2000))
    head.fit(X[~is_test], y[~is_test])
    metrics = evaluate(head, X[is_test], y[is_test])
    production = evaluate_production(extractor, X[is_test], y[is_test])
    with db.connect() as conn:
        # Newest photo this model has seen; drift checks compare later photos against it.
        data_cutoff = conn.execute("SELECT max(ingested_at) FROM photos").fetchone()[0]

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "extractor": extractor,
                "head": "StandardScaler+LogisticRegression",
                "head_C": HEAD_C,
                "confidence_threshold": config.CONFIDENCE_THRESHOLD,
                "n_train": int((~is_test).sum()),
                "n_test": int(is_test.sum()),
                "n_classes": len(head.classes_),
                "dataset_md5": dataset["md5"],
                "dataset_nfiles": dataset["nfiles"],
                "data_cutoff": data_cutoff.isoformat(),
            }
        )
        mlflow.set_tag("dvc.dataset_md5", dataset["md5"])
        mlflow.log_artifact(str(dataset["dvc_file"]), artifact_path="dataset")
        mlflow.log_metrics(metrics)
        if production:
            mlflow.set_tag("compared_with_version", production["version"])
            mlflow.log_metrics(
                {f"production_{k}": v for k, v in production.items() if k != "version"}
            )
        mlflow.log_dict(
            {
                "extractor": extractor,
                "classes": [{"taxon_id": int(t), **names[int(t)]} for t in head.classes_],
            },
            registry.META_ARTIFACT,
        )
        info = mlflow.sklearn.log_model(
            head, artifact_path="model", registered_model_name=config.MODEL_NAME
        )

    summary = {
        "run_id": run.info.run_id,
        "model_version": str(info.registered_model_version),
        "extractor": extractor,
        "dataset_md5": dataset["md5"],
        "n_train": int((~is_test).sum()),
        "n_test": int(is_test.sum()),
        **metrics,
        "production": production,
    }
    log.info("Run finished: %s", summary)
    if promote:
        summary["promotion"] = registry.promote_if_better(
            summary["run_id"], summary["model_version"]
        )
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(json.dumps(train(promote=True), indent=2))
