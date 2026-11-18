# Defense guide

Format: 15 minutes of presentation, 5 minutes of demonstration, 10 minutes of questions.
Every member must speak. The slides are the "Presentation" pages of the Streamlit app
(http://localhost:8501), so there is nothing else to open.

## Before the defense

Do this at least 30 minutes ahead. The first start downloads images and can be slow.

```bash
docker compose up -d
docker compose ps                    # all seven services up; api and mlflow "healthy"
```

Then check:

- [ ] http://localhost:8501 opens and slide 1 shows live figures (not the "unavailable" note)
- [ ] The "Production model" page shows a version number
- [ ] Airflow (http://localhost:8080) and Grafana (http://localhost:3000) are logged in
      (`admin` / `admin`), each in its own browser tab
- [ ] Grafana is on the "Species ID - API and model monitoring" dashboard, time range
      "Last 15 minutes", refresh 10 s
- [ ] MLflow (http://localhost:5001) is open on the `species-id` experiment
- [ ] Two test photos are ready on the desktop: one clear photo of a covered species
      (any file from `data/images/`), and one blurred or off-topic photo
- [ ] A terminal is open in the project folder, font size increased
- [ ] Send some normal traffic so the dashboard is not empty:
      `uv run python scripts/simulate_traffic.py --n 60`
- [ ] Laptop on power, notifications off, screen sharing tested

If the "High human-review rate" alert fired in the last 4 hours, Grafana will not notify
again for the same alert during that window. Restart Grafana to reset it:
`docker compose restart grafana`.

## Presentation: 15 minutes

Split the three blocks between members. With two members, one takes blocks A and C, the other
takes block B and drives the demo.

| Block | Slide | Minutes | Say |
|---|---|---|---|
| A | 1. Introduction | 1 | The service in one sentence. The project is about the production chain, not model performance. |
| A | 2. Objectives and scope | 1.5 | Level 1 of the brief: ten species, fixed extractor, full chain closed. Name the four key metrics. |
| A | 3. Project progress | 1.5 | Three phases, each with what was delivered. Be upfront about the optional items not done. |
| A | 4. Data | 2 | iNaturalist, research grade, open licenses only, attribution stored. Database, no CSV. DVC hash per training. |
| B | 5. Model | 1.5 | Frozen extractor + logistic regression. Explain why it helps MLOps: cached features, retraining in seconds. |
| B | 6. Architecture | 2.5 | Walk the diagram left to right. Why each service is separate. One command to start. |
| B | 7. Training pipeline | 2.5 | The six steps. Retraining only when needed. Fair comparison on the same test set. Show the version table. |
| C | 8. Prediction and monitoring | 2.5 | Real-time path. Two watchdogs: Evidently for inputs, Grafana for model behaviour. Both can trigger training. |

Slide 9 opens the demonstration; slide 10 closes after it.

## Demonstration: 5 minutes

The Grafana alert needs about two minutes to fire, so start it first.

| Time | Action | What to say |
|---|---|---|
| 0:00 | Terminal: `uv run python scripts/simulate_traffic.py --n 100 --drift` | "We now send 100 degraded photos: grey, blurred, dark. This simulates a camera problem or a change in what users submit." |
| 0:20 | Streamlit, **Identify**: upload the clear photo | "A normal photo: confident answer, served by the production model." |
| 1:00 | **Identify**: upload the blurred or off-topic photo | "Low confidence: the app refers it to human review instead of guessing." |
| 1:40 | Terminal: read the script output | "About 70% of the degraded photos were flagged for review." |
| 2:00 | Grafana dashboard | "Review rate jumps, mean confidence drops. The alert is pending, then firing." (Alerting → Alert rules shows the state.) |
| 3:00 | Airflow: `training_pipeline` | "A new run started by the Grafana webhook. Trigger reason is in the `retraining_needed` log. Steps: ingest, features, drift check, train, promote." |
| 4:00 | MLflow: `species-id` experiment, then Models | "New version, with its metrics, the DVC dataset hash, and the production alias." Then `species-id-monitoring` → open a drift report. |
| 4:40 | Streamlit, **Production model** | "The API already serves the new version, without a restart." |

Optional, if time allows: trigger the `prediction_drift` DAG in Airflow to show Evidently
detecting the same degraded photos (drift share 100%) and triggering training itself.

### If something goes wrong

| Problem | Fallback |
|---|---|
| Alert does not fire in time | Trigger `prediction_drift` manually in Airflow; it detects the drift and starts training. |
| Training run is slow | Show a previous completed run in Airflow and its version in MLflow. |
| iNaturalist API is unreachable | The ETL step retries; meanwhile show an earlier run. Training itself does not need the internet. |
| A service is down | `docker compose ps`, then `docker compose up -d`. Explain the healthchecks while it restarts. |

Then show slide 10 (results, limits, next steps) for 30 seconds before questions.

## Likely questions

**Why such a simple model?**
The brief asks for it, and it is evaluated on MLOps. A frozen extractor with a linear head
reaches about 94% accuracy on ten species and retrains in seconds on a CPU, which makes
frequent automated retraining realistic. The extractor is one setting; swapping it needs no
code change.

**How do you decide to put a model in production?**
At the end of training, the current production model is re-scored on the same test set as
the new model. Airflow reads both macro-F1 scores from MLflow and moves the `production`
alias unless the new model is worse by more than 0.005. Each version is tagged `promoted` or
`rejected`. The tolerance exists because differences that small are noise on a test set of a
few hundred photos, and the newer model has seen more recent data.

**Why macro-F1 rather than accuracy?**
It weighs every species equally, so a model cannot look good by being right only on the
species with the most photos. That matters as soon as the classes become imbalanced.

**The test set changes over time. Is the comparison fair?**
Yes for the promotion decision, because both models are scored on the same test set at the
same moment. Scores from different runs are not directly comparable, which is why the
production score is logged again on each run.

**How does drift detection work on images?**
On the extractor's feature vectors, not on pixels. The 1,280 dimensions are projected onto
10 PCA axes fitted on the reference sample, and each axis is tested with Kolmogorov-Smirnov
(p < 0.05). Drift is declared when at least half of the axes have shifted.

**Why not Evidently's default test?**
For large reference samples it uses a Wasserstein-distance threshold, which does not account
for the size of the current sample. With batches of about 80 predictions it flagged 40% of
features on perfectly normal photos. Kolmogorov-Smirnov gave 10% on normal photos and 100%
on degraded ones.

**If predictions drift, how does retraining help? You have no labels for them.**
It only helps partly, and this is a known limit. Retraining pulls fresh labelled photos from
iNaturalist, so the model follows the source distribution, but it does not learn from the
drifted photos themselves. The proper fix is the feedback loop of level 3: collect validated
identifications for the predictions and add them to training.

**What is versioned, and how do you reproduce a model?**
Code in Git; data with DVC (a hash of the image directory, logged in each MLflow run with the
`.dvc` file); models, parameters and metrics in MLflow. A run tells you which data hash,
which extractor and which settings produced a model.

**Why DVC without Git?**
Mentor's instruction. DVC runs with `--no-scm` inside the data folder, and MLflow keeps the
link between a model and its dataset hash, which Git commits would otherwise provide.

**Where is the data stored? Why not everything in the database?**
Metadata, license, attribution, features and predictions are in Postgres. Image files stay
on disk and the database stores their relative path, which is the usual practice: databases
handle large binary files poorly.

**How do you handle licenses?**
The ingestion query asks only for allowed licenses, then each photo is checked again
(license code and host) before download. License and attribution are stored per photo, so
any image can be traced and credited. A unit test covers this filter.

**Why does Airflow call the API to train instead of training itself?**
Separation of roles. The API image is the only one with PyTorch. Airflow stays light: it
schedules, runs ingestion and drift checks, and makes the promotion decision.

**What happens to requests during retraining?**
Nothing changes for users. The API keeps serving the current production model; it checks the
registry every 30 seconds and switches when the alias moves.

**What about an out-of-scope photo, such as a cat?**
The model can only answer with one of its ten species. Usually confidence is low and the
photo goes to human review, but that is not guaranteed. Proper out-of-scope detection is a
level 4 item and listed in next steps.

**How would you scale this?**
The API is stateless, so it can run as several replicas behind a load balancer, on
Kubernetes for example. Training would move to its own worker instead of the API. Images
would go to object storage with a DVC remote, and Postgres to a managed service.

**What would you do with more time?**
The feedback loop on real predictions, out-of-scope detection, a comparison of extractors
against their serving cost, API authentication, and a release pipeline to a registry.

**What was hardest?**
Good candidates to mention: making drift detection reliable on small batches; sharing data
paths between host and containers; version conflicts (MLflow with SQLAlchemy, Evidently
inside the Airflow image); and deciding what "retrain if necessary" should mean.

## Numbers worth knowing

Read the current values on the slides before the defense; these are orders of magnitude.

- 10 species, about 4,400 photos, one in five held out for testing
- About 94% test accuracy, macro-F1 about 0.94
- About 5% of test photos referred to human review; about 96% accuracy on the rest
- 7 services, 2 Airflow DAGs, 4 Grafana alerts, 20 unit tests
- Confidence threshold 0.6; promotion tolerance 0.005; drift when half of 10 axes shift
