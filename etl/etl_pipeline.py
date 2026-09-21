"""
etl_pipeline.py -- ingests RAW data into the system's Postgres tables.

This is separate from db.py:
  - db.py is called SYNCHRONOUSLY by the live API (app.py) for real-time
    events as they happen (a job gets shown/hired/rated right now).
  - etl_pipeline.py is a BATCH job for ingesting data that arrives from
    outside the live request path: bulk worker onboarding CSVs from
    partner agencies, legacy-system exports, analytics exports, daily
    dumps of raw app-event logs (JSONL) that weren't written directly
    via the API, etc.

Where raw files land:  data/raw/workers/*.csv, data/raw/jobs/*.csv,
                        data/raw/events/*.jsonl
After successful load:  moved to data/processed/<same subfolder>/
On failure:              moved to data/failed/<same subfolder>/, logged

Trigger this the same way as retrain_pipeline.py -- cron, e.g.:
    */15 * * * *  /usr/bin/python3 /opt/app/etl/etl_pipeline.py >> logs/etl.log 2>&1
(every 15 min is reasonable for near-real-time bulk ingestion; adjust to your volume)
"""

import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras
import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FAILED_DIR = BASE_DIR / "data" / "failed"
LOG_DIR = BASE_DIR / "logs"
DB_DSN = os.getenv("DATABASE_URL")

for d in [RAW_DIR / "workers", RAW_DIR / "jobs", RAW_DIR / "events",
          PROCESSED_DIR / "workers", PROCESSED_DIR / "jobs", PROCESSED_DIR / "events",
          FAILED_DIR / "workers", FAILED_DIR / "jobs", FAILED_DIR / "events", LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "etl.log"), logging.StreamHandler()],
)
log = logging.getLogger("etl")

REQUIRED_WORKER_COLS = {"worker_id", "name", "city", "lat", "lon", "category",
                         "age", "join_date", "hourly_rate_pkr"}
REQUIRED_JOB_COLS = {"job_id", "customer_id", "category", "lat", "lon",
                      "urgency", "budget_pkr"}
VALID_CATEGORIES = {"Plumbing", "Carpentry", "Electrician", "Mechanic", "Masonry"}
VALID_URGENCY = {"low", "medium", "high", "emergency"}


def get_conn():
    return psycopg2.connect(DB_DSN)


# ---------------------------------------------------------------------------
# WORKERS ingestion
# ---------------------------------------------------------------------------
def transform_workers(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_WORKER_COLS - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    before = len(df)
    df = df.dropna(subset=["worker_id", "lat", "lon", "category"])
    df = df[df["category"].isin(VALID_CATEGORIES)]
    df = df.drop_duplicates(subset=["worker_id"], keep="last")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    # sanity-check coordinates are roughly within Pakistan's bounding box
    df = df[df["lat"].between(23, 38) & df["lon"].between(60, 78)]
    df["hourly_rate_pkr"] = pd.to_numeric(df["hourly_rate_pkr"], errors="coerce").fillna(0).astype(int)
    df["is_verified"] = df.get("is_verified", False)

    # skills is optional and semicolon-delimited, e.g. "engine repair;oil change".
    # No category-specific allow-list -- whatever skills a worker lists are
    # accepted as-is and become filterable/rankable in the live service.
    def parse_skills(raw):
        if pd.isna(raw) or not str(raw).strip():
            return []
        return [s.strip() for s in str(raw).split(";") if s.strip()]

    if "skills" in df.columns:
        df["skills"] = df["skills"].apply(parse_skills)
    else:
        df["skills"] = [[] for _ in range(len(df))]

    dropped = before - len(df)
    if dropped:
        log.warning(f"workers: dropped {dropped}/{before} invalid rows during transform")
    return df


def load_workers(conn, df: pd.DataFrame) -> int:
    rows = 0
    with conn.cursor() as cur:
        for _, r in df.iterrows():
            cur.execute("""
                INSERT INTO workers (worker_id, name, city, location, category, age,
                                      join_date, hourly_rate_pkr, is_verified, skills)
                VALUES (%s,%s,%s, ST_SetSRID(ST_MakePoint(%s,%s),4326)::geography,
                        %s,%s,%s,%s,%s,%s)
                ON CONFLICT (worker_id) DO UPDATE SET
                    name = EXCLUDED.name, city = EXCLUDED.city, location = EXCLUDED.location,
                    category = EXCLUDED.category, hourly_rate_pkr = EXCLUDED.hourly_rate_pkr,
                    is_verified = EXCLUDED.is_verified, skills = EXCLUDED.skills, updated_at = now();
            """, (r["worker_id"], r["name"], r["city"], r["lon"], r["lat"],
                  r["category"], int(r["age"]) if pd.notna(r.get("age")) else None,
                  r["join_date"], r["hourly_rate_pkr"], bool(r["is_verified"]), r["skills"]))
            # ensure a worker_stats row exists (idempotent, defaults to zero/unknown)
            cur.execute("""
                INSERT INTO worker_stats (worker_id) VALUES (%s)
                ON CONFLICT (worker_id) DO NOTHING;
            """, (r["worker_id"],))
            rows += 1
    conn.commit()
    return rows


# ---------------------------------------------------------------------------
# JOBS ingestion
# ---------------------------------------------------------------------------
def transform_jobs(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_JOB_COLS - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    before = len(df)
    df = df.dropna(subset=["job_id", "lat", "lon", "category"])
    df = df[df["category"].isin(VALID_CATEGORIES)]
    df["urgency"] = df["urgency"].where(df["urgency"].isin(VALID_URGENCY), "medium")
    df = df.drop_duplicates(subset=["job_id"], keep="last")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    df["budget_pkr"] = pd.to_numeric(df["budget_pkr"], errors="coerce").fillna(0).astype(int)

    # required_skill is optional and accepted as-is -- if it doesn't match
    # any worker's skills for this category, the live service's fallback in
    # get_candidate_workers just widens to the full category automatically.
    def clean_skill(raw):
        if pd.isna(raw) or not str(raw).strip():
            return None
        return str(raw).strip()

    if "required_skill" in df.columns:
        df["required_skill"] = df["required_skill"].apply(clean_skill)
    else:
        df["required_skill"] = None

    dropped = before - len(df)
    if dropped:
        log.warning(f"jobs: dropped {dropped}/{before} invalid rows during transform")
    return df


def load_jobs(conn, df: pd.DataFrame) -> int:
    rows = 0
    with conn.cursor() as cur:
        for _, r in df.iterrows():
            cur.execute("""
                INSERT INTO jobs (job_id, customer_id, category, location, urgency, budget_pkr, required_skill)
                VALUES (%s,%s,%s, ST_SetSRID(ST_MakePoint(%s,%s),4326)::geography, %s,%s,%s)
                ON CONFLICT (job_id) DO NOTHING;
            """, (r["job_id"], r["customer_id"], r["category"], r["lon"], r["lat"],
                  r["urgency"], r["budget_pkr"], r["required_skill"]))
            rows += 1
    conn.commit()
    return rows


# ---------------------------------------------------------------------------
# RAW EVENT LOGS ingestion (JSONL -- e.g. from an analytics pipeline instead
# of the live API's direct db.py writes)
# ---------------------------------------------------------------------------
def load_events_jsonl(conn, path: Path) -> int:
    rows = 0
    with open(path) as f, conn.cursor() as cur:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                cur.execute("""
                    INSERT INTO match_events
                        (job_id, worker_id, event_type, distance_km, rank_position,
                         model_score, rating_smoothed_at_time, jobs_completed_at_time,
                         tenure_days_at_time, skill_match_at_time, post_job_rating, model_version)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
                """, (
                    e["job_id"], e["worker_id"], e["event_type"],
                    e.get("distance_km"), e.get("rank_position"), e.get("model_score"),
                    e.get("rating_smoothed_at_time"), e.get("jobs_completed_at_time"),
                    e.get("tenure_days_at_time"), e.get("skill_match_at_time"),
                    e.get("post_job_rating"), e.get("model_version"),
                ))
                rows += 1
            except (KeyError, json.JSONDecodeError) as ex:
                log.error(f"skipping malformed event line in {path.name}: {ex}")
    conn.commit()
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def process_folder(subfolder: str, extract_fn, transform_fn, load_fn, conn):
    src_dir = RAW_DIR / subfolder
    files = sorted(src_dir.glob("*.csv"))
    for f in files:
        try:
            df = extract_fn(f)
            df = transform_fn(df)
            n = load_fn(conn, df)
            dest = PROCESSED_DIR / subfolder / f"{datetime.utcnow():%Y%m%d_%H%M%S}_{f.name}"
            shutil.move(str(f), str(dest))
            log.info(f"[{subfolder}] loaded {n} rows from {f.name} -> archived as {dest.name}")
        except Exception as ex:
            dest = FAILED_DIR / subfolder / f.name
            shutil.move(str(f), str(dest))
            log.error(f"[{subfolder}] FAILED on {f.name}: {ex} -> moved to failed/")


def process_events(conn):
    src_dir = RAW_DIR / "events"
    for f in sorted(src_dir.glob("*.jsonl")):
        try:
            n = load_events_jsonl(conn, f)
            dest = PROCESSED_DIR / "events" / f"{datetime.utcnow():%Y%m%d_%H%M%S}_{f.name}"
            shutil.move(str(f), str(dest))
            log.info(f"[events] loaded {n} rows from {f.name} -> archived as {dest.name}")
        except Exception as ex:
            dest = FAILED_DIR / "events" / f.name
            shutil.move(str(f), str(dest))
            log.error(f"[events] FAILED on {f.name}: {ex} -> moved to failed/")


def main():
    log.info("=== ETL run starting ===")
    conn = get_conn()
    try:
        process_folder("workers", pd.read_csv, transform_workers, load_workers, conn)
        process_folder("jobs", pd.read_csv, transform_jobs, load_jobs, conn)
        process_events(conn)
    finally:
        conn.close()
    log.info("=== ETL run complete ===")


if __name__ == "__main__":
    main()