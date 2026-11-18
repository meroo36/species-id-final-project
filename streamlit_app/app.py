"""Streamlit front end: the application and the defense presentation.

Run with:  uv run streamlit run streamlit_app/app.py

The "Application" pages talk to the API to identify photos. The "Presentation" pages are the
slides for the defense; their figures are read live from the API so they never go stale.
"""

import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Species ID", page_icon="🔎", layout="wide")


# --------------------------------------------------------------------------- API access


@st.cache_data(ttl=20)
def api_get(path: str) -> dict | None:
    try:
        response = httpx.get(f"{API_URL}{path}", timeout=15)
        return response.json() if response.status_code == 200 else None
    except httpx.HTTPError:
        return None


def identify(name: str, content: bytes) -> dict:
    response = httpx.post(f"{API_URL}/predict", files={"file": (name, content)}, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(response.json().get("detail", response.text))
    return response.json()


def api_missing() -> None:
    st.info("Live figures are unavailable: the API is not reachable or has no model yet.")


# --------------------------------------------------------------------------- diagrams

GRAPH_STYLE = """
    graph [rankdir=LR, bgcolor="transparent", pad=0.2, nodesep=0.35, ranksep=0.55]
    node  [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,
           color="#5b6b7b", fillcolor="#eef2f6", fontcolor="#1f2a36"]
    edge  [fontname="Helvetica", fontsize=9, color="#5b6b7b", fontcolor="#5b6b7b"]
"""

ARCHITECTURE = f"""digraph {{ {GRAPH_STYLE}
    user [label="User", shape=ellipse, fillcolor="#ffffff"]
    inat [label="iNaturalist\\nopen data", shape=cylinder, fillcolor="#ffffff"]
    streamlit [label="Streamlit\\nweb app", fillcolor="#fde2e2"]
    api [label="FastAPI\\nmodel API", fillcolor="#d9f2e6"]
    airflow [label="Airflow\\npipelines", fillcolor="#fde9d4"]
    postgres [label="Postgres", shape=cylinder, fillcolor="#e3e0f8"]
    mlflow [label="MLflow\\ntracking + registry", fillcolor="#d8ebfa"]
    prometheus [label="Prometheus", fillcolor="#fbe6c8"]
    grafana [label="Grafana\\ndashboard + alerts", fillcolor="#fbe6c8"]

    user -> streamlit
    streamlit -> api [label="/predict"]
    inat -> airflow [label="new photos"]
    airflow -> api [label="/features\\n/training"]
    airflow -> postgres [label="metadata\\ndrift checks"]
    api -> postgres [label="features\\npredictions"]
    api -> mlflow [label="runs, models"]
    airflow -> mlflow [label="promotion\\ndrift reports"]
    prometheus -> api [label="scrapes\\n/metrics"]
    grafana -> prometheus
    grafana -> api [label="alert webhook", style=dashed]
    api -> airflow [label="trigger DAG", style=dashed]
}}"""

TRAINING_PIPELINE = f"""digraph {{ {GRAPH_STYLE}
    etl [label="1. Extract,\\ntransform, load", fillcolor="#fde9d4"]
    feat [label="2. Compute\\nfeatures", fillcolor="#d9f2e6"]
    drift [label="3. Check data drift\\n(Evidently)", fillcolor="#fde9d4"]
    need [label="4. Retraining\\nneeded?", shape=diamond, fillcolor="#ffffff"]
    stop [label="Stop", shape=ellipse, fillcolor="#ffffff"]
    train [label="5. Train and\\nevaluate", fillcolor="#d9f2e6"]
    prod [label="6. Determine\\nproduction model", fillcolor="#fde9d4"]
    mlflow [label="MLflow", fillcolor="#d8ebfa"]

    etl -> feat -> drift -> need
    need -> stop [label="no"]
    need -> train [label="yes"]
    train -> prod
    drift -> mlflow [label="report", style=dashed]
    train -> mlflow [label="run, model,\\ndataset hash", style=dashed]
    prod -> mlflow [label="production tag", style=dashed]
}}"""

PREDICTION_PIPELINE = f"""digraph {{ {GRAPH_STYLE}
    photo [label="Photo", shape=ellipse, fillcolor="#ffffff"]
    api [label="POST /predict", fillcolor="#d9f2e6"]
    model [label="Production model\\n(MLflow registry)", fillcolor="#d8ebfa"]
    answer [label="Species, confidence,\\nreview flag", shape=ellipse, fillcolor="#ffffff"]
    store [label="Store prediction\\n+ features", shape=cylinder, fillcolor="#e3e0f8"]
    metrics [label="Prometheus\\nmetrics", fillcolor="#fbe6c8"]
    pdrift [label="Hourly drift check\\n(Evidently, Airflow)", fillcolor="#fde9d4"]
    alerts [label="Grafana alerts", fillcolor="#fbe6c8"]
    retrain [label="Training\\npipeline", fillcolor="#fde9d4"]

    photo -> api -> answer
    model -> api
    api -> store -> pdrift
    api -> metrics -> alerts
    pdrift -> retrain [label="drift"]
    alerts -> retrain [label="webhook"]
}}"""


# --------------------------------------------------------------------------- application


def page_identify() -> None:
    st.title("Identify a species")
    st.caption("Upload a photograph. Uncertain results are referred to human review.")
    left, right = st.columns([1, 1], gap="large")
    upload = left.file_uploader("Photo", type=["jpg", "jpeg", "png", "webp"])
    if not upload:
        return
    left.image(upload, use_container_width=True)
    try:
        with st.spinner("Identifying..."):
            result = identify(upload.name, upload.getvalue())
    except (RuntimeError, httpx.HTTPError) as error:
        right.error(f"The API could not identify this photo: {error}")
        return
    best = result["prediction"]
    common = best["common_name"] or best["scientific_name"]
    if result["needs_review"]:
        right.warning(
            f"Best guess: **{common}** (*{best['scientific_name']}*), but confidence is only "
            f"{result['confidence']:.0%}. This photo is referred to human review."
        )
    else:
        right.success(
            f"**{common}** (*{best['scientific_name']}*), confidence {result['confidence']:.0%}."
        )
    right.subheader("Top candidates")
    for candidate in result["top"]:
        label = candidate["common_name"] or candidate["scientific_name"]
        right.progress(
            min(candidate["confidence"], 1.0),
            text=f"{label} ({candidate['scientific_name']}): {candidate['confidence']:.1%}",
        )
    right.caption(f"Served by model version {result['model_version']}.")


def page_model() -> None:
    st.title("Production model")
    model = api_get("/model")
    if model is None:
        api_missing()
        return
    metrics, params = model["metrics"], model["params"]
    columns = st.columns(6)
    columns[0].metric("Version", model["version"])
    columns[1].metric("Test accuracy", f"{metrics['accuracy']:.1%}")
    columns[2].metric("Macro-F1", f"{metrics['macro_f1']:.3f}")
    columns[3].metric("Training photos", params["n_train"])
    columns[4].metric("Test photos", params["n_test"])
    columns[5].metric("Human-review rate", f"{metrics['review_rate']:.1%}")
    st.write(f"**Extractor:** `{model['extractor']}`")
    st.write(f"**Dataset version (DVC):** `{params['dataset_md5']}`")
    st.subheader(f"Species covered ({len(model['classes'])})")
    st.dataframe(
        [
            {
                "Common name": c["common_name"],
                "Scientific name": c["scientific_name"],
                "iNaturalist taxon": c["taxon_id"],
            }
            for c in model["classes"]
        ],
        hide_index=True,
        use_container_width=True,
    )


# --------------------------------------------------------------------------- presentation


def slide_title() -> None:
    st.title("Species identification from photographs")
    st.subheader("An automated, monitored, retrainable MLOps pipeline")
    st.write("")
    st.markdown(
        """
**The service.** A user submits a photograph and gets a species with a confidence level.
When the model is not sure enough, the photo is referred to a human.

**The project.** The focus is the production chain, not raw model performance:

> A simple model that is versioned, monitored, deployed and automatically retrainable is
> worth more than a sophisticated model left in a notebook.
"""
    )
    stats, model = api_get("/stats"), api_get("/model")
    if stats and model:
        st.write("")
        columns = st.columns(5)
        columns[0].metric("Species", len(stats["species"]))
        columns[1].metric("Licensed photos", sum(s["photos"] for s in stats["species"]))
        columns[2].metric("Model versions", len(stats["versions"]))
        columns[3].metric("Test accuracy", f"{model['metrics']['accuracy']:.1%}")
        columns[4].metric("Services", 7)


def slide_objectives() -> None:
    st.title("Objectives and scope")
    left, right = st.columns(2, gap="large")
    left.subheader("What we set out to build")
    left.markdown(
        """
- Identify a species from a photo, with a confidence level
- Refer uncertain photos to **human review**
- Use **only openly licensed** photos, and keep their attribution
- Close the loop: ingestion → training → deployment → monitoring → retraining,
  **with no manual step**
"""
    )
    right.subheader("Scope: level 1 of the brief")
    right.markdown(
        """
- About ten well-represented species
- A fixed pre-trained extractor
- The full chain, effectively closed

**Key metrics**

| Metric | Target |
|---|---|
| Macro-F1 on the test set | ≥ 0.90 |
| Share of photos sent to review | < 10% |
| Accuracy when the model is confident | ≥ 0.95 |
| Prediction latency (p95) | < 2 s |
"""
    )


def slide_roadmap() -> None:
    st.title("Project progress")
    one, two, three = st.columns(3, gap="large")
    one.subheader("1. Foundations")
    one.markdown(
        """
- Postgres database
- One-time ingestion script
- Baseline model: `training.py`, `predict.py`
- FastAPI: `/training`, `/predict`
"""
    )
    two.subheader("2. Microservices, tracking, versioning")
    two.markdown(
        """
- MLflow tracking and Model Registry
- Compare with production, tag the best
- DVC dataset hash in every run
- Docker Compose microservices
- Scheduled Airflow pipeline
- Unit tests and CI
"""
    )
    three.subheader("3. Monitoring and maintenance")
    three.markdown(
        """
- Evidently drift checks in Airflow
- Predictions stored with features
- Prometheus metrics, Grafana dashboard
- Alerts, with a webhook that retrains
- This Streamlit application
"""
    )
    st.divider()
    st.markdown(
        "**Not done (optional items):** release workflow to Docker Hub, API authentication, "
        "Kubernetes."
    )


def slide_data() -> None:
    st.title("Data")
    left, right = st.columns([1, 1], gap="large")
    left.markdown(
        """
**Source.** The iNaturalist API: citizen-science observations, validated by the community.

**What the pipeline keeps**

- *Research-grade* observations only
- Photos under an **allowed open license** (`cc0`, `cc-by`, `cc-by-nc`)
- Photos hosted on the **open-data bucket** only
- One photo per observation, to avoid near-duplicates across train and test

**What is stored.** Files on disk; metadata, **license and attribution** in Postgres.
No CSV files anywhere.

**Split.** Deterministic, by observation id: one observation in five is test.

**Versioning.** DVC hashes the image directory before each training; the hash is logged
to MLflow with the run.
"""
    )
    stats = api_get("/stats")
    if stats is None:
        with right:
            api_missing()
        return
    right.subheader("Photos per species (live)")
    right.dataframe(
        [
            {
                "Species": s["common_name"] or s["scientific_name"],
                "Scientific name": s["scientific_name"],
                "Photos": s["photos"],
                "Test": s["test_photos"],
            }
            for s in stats["species"]
        ],
        hide_index=True,
        use_container_width=True,
    )
    right.subheader("Licenses (live)")
    right.bar_chart(
        {row["license"]: row["photos"] for row in stats["licenses"]}, horizontal=True, height=160
    )


def slide_model() -> None:
    st.title("Model")
    left, right = st.columns([1, 1], gap="large")
    left.markdown(
        """
**Deliberately simple**, as the brief recommends: at most a day or two on modelling.

1. A **frozen pre-trained extractor** (`mobilenetv3_large_100`, via `timm`) turns each
   photo into a 1,280-number feature vector.
2. A **logistic-regression head** classifies the vector.

**Why this design helps MLOps**

- Features are computed once per photo and **cached in Postgres**
- Retraining only refits the head: **seconds on a CPU**, no GPU
- The same vectors feed **drift detection**
- Swapping the extractor is one setting

**Human review.** Below a confidence of 0.6, the answer is flagged `needs_review`.
"""
    )
    model = api_get("/model")
    if model is None:
        with right:
            api_missing()
        return
    metrics = model["metrics"]
    right.subheader(f"Production model: version {model['version']} (live)")
    a, b = right.columns(2)
    a.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
    b.metric("Macro-F1", f"{metrics['macro_f1']:.3f}")
    a.metric("Sent to human review", f"{metrics['review_rate']:.1%}")
    b.metric("Accuracy when confident", f"{metrics['accuracy_when_confident']:.1%}")
    right.caption(
        f"Trained on {model['params']['n_train']} photos, tested on {model['params']['n_test']}."
    )


def slide_architecture() -> None:
    st.title("Architecture")
    st.graphviz_chart(ARCHITECTURE, use_container_width=True)
    st.markdown(
        """
| Service | Role | Why it is separate |
|---|---|---|
| **FastAPI** | Features, training, prediction, metrics | The only service with the heavy ML stack (PyTorch) |
| **Airflow** | Orchestrates ingestion, drift checks, promotion | Light image; schedules and decides, does not train |
| **MLflow** | Runs, artifacts, Model Registry | Single source of truth for which model is in production |
| **Postgres** | Application data + MLflow and Airflow metadata | One database server, three databases |
| **Prometheus + Grafana** | Metrics, dashboard, alerts | Provisioned from files: nothing is configured by hand |
| **Streamlit** | User interface | Talks only to the API |
"""
    )
    st.caption("Everything runs locally with one command: docker compose up -d --build")


def slide_training() -> None:
    st.title("Training pipeline")
    st.caption("Airflow DAG `training_pipeline`: daily, and on demand when monitoring asks.")
    st.graphviz_chart(TRAINING_PIPELINE, use_container_width=True)
    left, right = st.columns(2, gap="large")
    left.markdown(
        """
**Retraining happens only if needed**

- data drift was detected, or
- enough new photos have arrived (1,000), or
- there is no production model yet, or
- the run was forced by an alert or by prediction drift

**A fair comparison.** The production model is re-scored on the *same* test set as the new
one. The new model takes the `production` tag unless it is worse by more than 0.005 macro-F1.
"""
    )
    stats = api_get("/stats")
    if stats is None:
        with right:
            api_missing()
        return
    right.subheader("Model versions (live, from MLflow)")
    right.dataframe(
        [
            {
                "Version": v["version"],
                "Status": "production" if v["production"] else v["status"],
                "Macro-F1": v["macro_f1"],
                "Production macro-F1 at the time": v["production_macro_f1"],
                "Training photos": v["n_train"],
            }
            for v in stats["versions"]
        ],
        hide_index=True,
        use_container_width=True,
        column_config={
            "Macro-F1": st.column_config.NumberColumn(format="%.4f"),
            "Production macro-F1 at the time": st.column_config.NumberColumn(format="%.4f"),
        },
    )


def slide_prediction() -> None:
    st.title("Prediction pipeline and monitoring")
    st.graphviz_chart(PREDICTION_PIPELINE, use_container_width=True)
    left, right = st.columns(2, gap="large")
    left.markdown(
        """
**Two independent watchdogs**

| | Watches | Tool | Reaction |
|---|---|---|---|
| Input drift | Features of submitted photos | Evidently, hourly Airflow DAG | Triggers training |
| Model behaviour | Share of low-confidence answers | Grafana alert on Prometheus metrics | Webhook triggers training |

Other alerts: slow predictions (p95 > 2 s), server errors (> 5%), API down.

**How drift is measured.** Feature vectors are projected on 10 PCA axes; each axis is tested
with Kolmogorov-Smirnov. Drift means at least half of the axes have shifted.
"""
    )
    stats = api_get("/stats")
    if stats is None:
        with right:
            api_missing()
        return
    predictions = stats["predictions"]
    a, b = right.columns(2)
    a.metric("Predictions stored", predictions["total"])
    share = predictions["needs_review"] / predictions["total"] if predictions["total"] else 0
    b.metric("Referred to human review", f"{share:.0%}")
    right.subheader("Latest drift checks (live)")
    right.dataframe(
        [
            {
                "When": c["checked_at"].replace("T", " ")[:19],
                "Kind": c["kind"],
                "Sample": c["n_current"],
                "Drifted features": f"{c['drift_share']:.0%}",
                "Drift": "yes" if c["drift_detected"] else "no",
            }
            for c in stats["drift_checks"]
        ],
        hide_index=True,
        use_container_width=True,
    )


def slide_demo() -> None:
    st.title("Live demonstration")
    st.markdown(
        """
| Step | Where | What to show |
|---|---|---|
| 1 | **Identify** page (this app) | Upload a clear photo: confident answer |
| 2 | **Identify** page | Upload a blurred or off-topic photo: referred to human review |
| 3 | Terminal | `uv run python scripts/simulate_traffic.py --n 100 --drift` |
| 4 | [Grafana](http://localhost:3000) | Review rate climbs; the alert fires |
| 5 | [Airflow](http://localhost:8080) | A forced `training_pipeline` run appears and completes |
| 6 | [MLflow](http://localhost:5001) | New model version, dataset hash, drift report |
| 7 | **Production model** page | The API now serves the new version |
"""
    )
    st.info(
        "The alert needs about two minutes to fire after the degraded photos are sent. "
        "Send them at the start of the demo, then show steps 1 and 2 while waiting."
    )


def slide_conclusion() -> None:
    st.title("Results, limits and next steps")
    one, two, three = st.columns(3, gap="large")
    one.subheader("What works")
    one.markdown(
        """
- The loop is closed: data to deployment to retraining, unattended
- Every model is traceable to its data (DVC hash) and its run (MLflow)
- Only the best model serves, and it changes without a restart
- Two monitoring paths trigger retraining on their own
- License and attribution kept for every photo
"""
    )
    two.subheader("Limits")
    two.markdown(
        """
- Ten species only; unknown species are not detected as such
- Retraining after prediction drift uses new iNaturalist photos, not the drifted ones
  (no labels for those)
- Human review is a flag, not yet a review queue
- Runs locally; no cloud deployment or autoscaling
- API without authentication
"""
    )
    three.subheader("Next steps")
    three.markdown(
        """
- Feedback loop: collect validated identifications, measure real accuracy
- Out-of-scope detection (no organism, unknown taxon)
- Compare extractors: accuracy gain against serving cost
- More species, with class-balanced training
- Authentication, Kubernetes, release pipeline
"""
    )


# --------------------------------------------------------------------------- navigation

navigation = st.navigation(
    {
        "Presentation": [
            st.Page(slide_title, title="1. Introduction", url_path="intro", default=True),
            st.Page(slide_objectives, title="2. Objectives and scope", url_path="objectives"),
            st.Page(slide_roadmap, title="3. Project progress", url_path="progress"),
            st.Page(slide_data, title="4. Data", url_path="data"),
            st.Page(slide_model, title="5. Model", url_path="model-design"),
            st.Page(slide_architecture, title="6. Architecture", url_path="architecture"),
            st.Page(slide_training, title="7. Training pipeline", url_path="training"),
            st.Page(slide_prediction, title="8. Prediction and monitoring", url_path="prediction"),
            st.Page(slide_demo, title="9. Live demonstration", url_path="demo"),
            st.Page(slide_conclusion, title="10. Results and next steps", url_path="conclusion"),
        ],
        "Application": [
            st.Page(page_identify, title="Identify", url_path="identify"),
            st.Page(page_model, title="Production model", url_path="production-model"),
        ],
    }
)
navigation.run()
