# Worker-Job Matching Service

A recommendation microservice that matches blue-collar service jobs
(plumbing, carpentry, electrician, mechanic, masonry) to the right worker
in a Pakistani gig-marketplace app, with a built-in cold-start strategy.

**This project has been run end-to-end against a real Postgres+PostGIS
instance during development** — schema applied, ETL tested with real
inserts, the API tested with real geo-queries and event logging, and the
retrain gate tested against real (if small) data. It's not just sketched,
it's been verified to actually connect.

---

## Core idea

Two problems, one system:

1. **Ranking** — given a job, which of the nearby, category-matching
   workers is the best fit? This is a **learning-to-rank** problem, not
   plain classification: we care about the *order* of candidates, not
   just yes/no per worker. Solved with a LightGBM LambdaRank model over
   structured features (distance, rating, experience, reliability, price).

2. **Cold start** — brand-new workers have no history for the model to
   learn from. Solved with three layers working together, not one trick:
   - A **Bayesian-smoothed rating** so an unrated worker defaults to the
     platform average instead of looking broken (NaN) or looking bad (0).
   - A **rule-based fallback** that works from day one, before any model
     exists at all.
   - An **exploration bonus** (epsilon-greedy) that deliberately surfaces
     new workers sometimes, so the platform actually collects data on
     them instead of starving them of jobs forever. This layer never
     turns off — every new worker is cold-start on day one, permanently.

The system closes the loop: every recommendation is logged, every
outcome (hired/completed/rated) feeds back into Postgres, and a gated
retraining pipeline periodically improves the model **only if it
measurably beats the current one** on real held-out data.

---

## Architecture

```
                    ┌─────────────────────┐
  Raw data sources  │   etl/etl_pipeline.py │   (batch: partner CSVs, legacy exports,
  (CSV/JSONL files) │        (ETL)          │    raw analytics event dumps)
                    └──────────┬───────────┘
                               │ validates, cleans, upserts
                               ▼
                    ┌─────────────────────┐
                    │   Postgres + PostGIS  │  ◄── single source of truth
                    │  workers / jobs /      │      (db/schema.sql)
                    │  worker_stats /         │
                    │  match_events /          │
                    │  model_registry           │
                    └──────────┬───────────┘
                               │
              live reads/writes│ (synchronous, per-request)
                               ▼
                    ┌─────────────────────┐
   Your app/backend │   service/app.py      │  POST /jobs   -> candidate gen
   (mobile/web)   ──▶      (FastAPI)         │                  + ranking
                    │                        │  POST /feedback -> logs outcome
                    └──────────┬───────────┘
                               │ reads model file
                               ▼
                    ┌─────────────────────┐
                    │  ml/models/production.txt│  ◄── swapped atomically on promotion
                    └──────────▲───────────┘
                               │ writes, only if better
                    ┌─────────────────────┐
                    │  ml/retrain_pipeline.py │  (cron: nightly/weekly)
                    │  time-based split       │
                    │  NDCG eval gate          │
                    │  model_registry log      │
                    └─────────────────────┘
```

**Why two ingestion paths (db.py vs etl_pipeline.py)?**
`db.py` is called synchronously by the live API for events happening
*right now* (a job just got shown/hired/rated). `etl_pipeline.py` is a
separate batch job for data arriving *outside* that live request path —
bulk worker onboarding from a partner, a legacy-system export, or a raw
analytics event dump that wasn't written directly through the API. Same
destination tables, different entry point, because they have different
volume/latency/failure-handling needs.

---

## Folder structure

```
worker-matching-service/
├── README.md                 ← you are here
├── requirements.txt
├── .env.example
├── data/
│   ├── generate_data.py       synthetic Pakistani-market data generator (for testing only)
│   ├── synthetic/              synthetic workers.csv / jobs.csv / interactions.csv
│   ├── raw/                    DROP raw files here for ETL: raw/workers/*.csv, raw/jobs/*.csv, raw/events/*.jsonl
│   ├── processed/               ETL archives successfully-loaded files here
│   ├── failed/                   ETL moves bad files here, with an error logged
│   └── training-snapshots/        dated Parquet snapshots, written by retrain_pipeline.py
├── db/
│   ├── schema.sql              Postgres+PostGIS schema -- run this first
│   └── db.py                   connection + event-logging functions used by the live API
├── etl/
│   └── etl_pipeline.py         batch ingestion: raw/ -> Postgres, with validation + archiving
├── ml/
│   ├── train_ranker.py          one-off training script against synthetic data (experimentation only)
│   ├── retrain_pipeline.py       PRODUCTION retraining: real data, time-split, eval gate, promotion
│   └── models/
│       ├── worker_ranker.txt      fallback model (trained on synthetic data)
│       └── production.txt          live model -- created once retrain_pipeline.py promotes one
└── service/
    └── app.py                    FastAPI service: POST /jobs, POST /feedback, GET /health
```

---

## Setup — do this once

```bash
# 1. Install Postgres with PostGIS (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib postgis

# 2. Create the database
sudo -u postgres createdb matching
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"

# 3. Install Python dependencies
pip install -r requirements.txt --break-system-packages

# 4. Apply the schema
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/matching"
psql "$DATABASE_URL" -f db/schema.sql
```

---

## Procedure — the order to actually run things in

### Step 1 — Generate synthetic data (optional, for testing the pipeline only)
```bash
python3 data/generate_data.py
```
This writes `workers.csv` / `jobs.csv` / `interactions.csv` to `data/synthetic/`.
**Do not treat this as real training data** — it's structurally realistic
(right cities, right price bands) but the hire probabilities are simulated,
not real. Use it only to prove the pipeline plumbing works.

### Step 2 — Seed the database via ETL
Drop CSVs into `data/raw/workers/` and `data/raw/jobs/` (for a first run,
you can copy from `data/synthetic/` — jobs.csv needs a `customer_id`
column added first). Then:
```bash
python3 etl/etl_pipeline.py
```
Check `data/processed/` for successfully-loaded files and `logs/etl.log`
for a full run log. Files that fail validation land in `data/failed/`
instead of silently corrupting your database.

### Step 3 — Start the live service
```bash
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/matching"
uvicorn service.app:app --reload --port 8000
```
Test it:
```bash
curl -X POST http://localhost:8000/jobs -H "Content-Type: application/json" -d '{
  "customer_id": "C00042", "category": "Plumbing",
  "lat": 24.8607, "lon": 67.0011, "urgency": "high", "budget_pkr": 2500, "top_n": 5
}'
```
This is the trigger point: your app's backend calls `POST /jobs` the
moment a customer posts a job, and gets ranked workers back in the same
response. Every candidate shown is automatically logged to `match_events`.

### Step 4 — Wire in feedback as jobs progress
Call `POST /feedback` at each real-world milestone:
```bash
# customer confirms a worker
curl -X POST http://localhost:8000/feedback -d '{"job_id":"J-...","worker_id":"W...","event_type":"hired"}'
# job finishes
curl -X POST http://localhost:8000/feedback -d '{"job_id":"J-...","worker_id":"W...","event_type":"completed"}'
# customer rates it
curl -X POST http://localhost:8000/feedback -d '{"job_id":"J-...","worker_id":"W...","event_type":"rated","rating":4.8}'
```
This is what turns synthetic-quality data into real training data.

### Step 5 — Retrain, on a schedule, once real data accumulates
```bash
# crontab -e
0 2 * * *  cd /path/to/worker-matching-service && DATABASE_URL=postgresql://postgres:postgres@localhost:5432/matching python3 ml/retrain_pipeline.py >> logs/retrain.log 2>&1
```
This pulls real `match_events`, splits by time (not randomly), trains,
evaluates NDCG@3/@5 on the held-out recent window, and **only promotes
the new model if it measurably beats the current production one** —
recorded either way in `model_registry` for audit/rollback.

---

## What to actually trust, and when

`service/app.py` now enforces this automatically instead of leaving it as
a manual reminder: it blends the model's ranking with a transparent
rule-based score via `MODEL_TRUST_WEIGHT`, and picks a conservative
default based on which model is actually loaded.

| Stage | Active model | `MODEL_TRUST_WEIGHT` (default) | What that means |
|---|---|---|---|
| Launch / early | `ml/models/worker_ranker.txt` (synthetic-trained fallback) | **0.30** | Mostly rule-based (distance, verification, reliability); the model nudges rather than decides |
| Mature | `ml/models/production.txt` (promoted by `retrain_pipeline.py` on real data) | **0.65** | Model-led, rule-based still keeps a floor under it |
| Forever | either | n/a | Exploration bonus for cold-start workers stays on — new workers join every day |

Override with the `MODEL_TRUST_WEIGHT` env var once you've validated real
performance and want to shift the balance further — don't hardcode 1.0
even once you trust the model; the rule-based floor is cheap insurance
against a bad retrain slipping past the NDCG gate.

Don't manually copy or rename `worker_ranker.txt` to `production.txt` to
"unlock" the higher trust weight — the check only looks at the filename,
not what the model was actually trained on, so doing that would give a
still-synthetic model more trust than it's earned. Only let
`retrain_pipeline.py` write `production.txt`, since that's the only path
that's actually trained on real `match_events` and passed the NDCG gate.

---

## Known limitations to address before real production use

- **Auth/security**: the API has no authentication layer yet — add one
  before exposing it beyond localhost.
- **Connection pooling**: `db.py` opens a new connection per request;
  fine for testing, replace with a connection pool (e.g. `psycopg2.pool`
  or `asyncpg`) under real load.
- **`data-analysis` observability**: track hire rate, time-to-hire, and
  cold-start worker exposure rate as business metrics — NDCG alone
  doesn't tell you if the *business* is healthy.
- **Radius/candidate-set tuning**: `radius_km` and the 50-candidate cap
  in `get_candidate_workers` are starting defaults — tune against your
  actual worker density per city.
