"""
Worker-Job Matching Microservice.
Run from project root:  uvicorn service.app:app --reload --port 8000

Workers/jobs are read from and written to Postgres directly (via db.py) --
NOT from a CSV file. The synthetic CSVs are only used to seed Postgres via
the ETL pipeline for local development; the live service always queries
the real database so it stays consistent with whatever ETL/onboarding adds.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Literal
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from datetime import date
import uuid
import sys
import os

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from db.db import get_conn, create_job, get_candidate_workers, log_shown, log_hired, log_completed, log_cancelled, log_rating

app = FastAPI(title="Worker Matching Service")

MODEL_PATH = BASE_DIR / "ml" / "models" / "production.txt"
FALLBACK_MODEL_PATH = BASE_DIR / "ml" / "models" / "worker_ranker.txt"
active_model_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL_PATH
model = lgb.Booster(model_file=str(active_model_path))

# How much to trust the LEARNED model's ranking vs. a transparent rule-based
# score. This is the codified version of the "trust ladder" from the README:
# low trust while we're only running on the SYNTHETIC-trained fallback model
# (its weights were fit to a simulated utility function, not real behavior),
# higher trust once retrain_pipeline.py has promoted a model trained on real
# match_events. Override via env var once you've validated real performance.
USING_REAL_MODEL = active_model_path.name == "production.txt"
DEFAULT_TRUST_WEIGHT = 0.65 if USING_REAL_MODEL else 0.30
MODEL_TRUST_WEIGHT = float(os.environ.get("MODEL_TRUST_WEIGHT", DEFAULT_TRUST_WEIGHT))

FEATURES = [
    "distance_km", "budget_pkr", "worker_hourly_rate_pkr",
    "worker_rating_smoothed", "worker_num_reviews", "worker_jobs_completed",
    "worker_tenure_days", "worker_is_verified", "worker_response_time_min",
    "worker_cancellation_rate", "urgency", "skill_match"
]
URGENCY_MAP = {"low": 0, "medium": 1, "high": 2, "emergency": 3}
DEFAULT_RESPONSE_TIME_MIN = 45.0  # conservative assumption for a worker with no track record yet


def smoothed_rating(avg_rating, num_reviews, prior=4.2, prior_weight=5):
    if avg_rating is None or num_reviews == 0:
        return prior
    return (prior * prior_weight + float(avg_rating) * num_reviews) / (prior_weight + num_reviews)


def rule_based_score(row) -> float:
    """Transparent, zero-training-data scoring function. Same signal the
    model uses, but with hand-set weights instead of learned ones -- this
    is the score we lean on MORE while the model is still synthetic-trained.

    skill_match gets the single biggest weight: a mechanic who does painting
    but not engine repair is a worse match than a slightly farther/pricier
    mechanic who does have the skill, so subcategory fit should dominate
    proximity/rating rather than just nudge it."""
    proximity = 1.0 / (1.0 + row["distance_km"])
    rating_component = (row["worker_rating_smoothed"] - 1) / 4.0        # -> 0..1
    reliability = 1.0 - min(row["worker_cancellation_rate"], 1.0)
    verified = float(row["worker_is_verified"])
    experience = min(np.log1p(row["worker_jobs_completed"]) / np.log1p(50), 1.0)
    skill = float(row["skill_match"])
    return (
        0.30 * skill
        + 0.28 * proximity
        + 0.18 * rating_component
        + 0.11 * reliability
        + 0.07 * verified
        + 0.06 * experience
    )


class JobRequest(BaseModel):
    customer_id: str = "guest"
    category: str
    lat: float
    lon: float
    urgency: str = "medium"
    budget_pkr: int = 2000
    radius_km: float = 25.0
    top_n: int = 5
    required_skill: Optional[str] = None  # e.g. "engine repair", "denting/painting"


class RankedWorker(BaseModel):
    worker_id: str
    name: str
    score: float
    distance_km: float
    rating_smoothed: float
    is_cold_start: bool
    has_required_skill: bool


class JobResponse(BaseModel):
    job_id: str
    recommendations: List[RankedWorker]


class FeedbackRequest(BaseModel):
    job_id: str
    worker_id: str
    event_type: Literal["hired", "completed", "cancelled", "rated"]
    rating: Optional[float] = None  # required when event_type == "rated"


VALID_CATEGORIES = {"Plumbing", "Carpentry", "Electrician", "Mechanic", "Masonry"}


def build_feature_frame(candidates: list, job: JobRequest) -> pd.DataFrame:
    rows = []
    today = date.today()
    for w in candidates:
        tenure_days = (today - w["join_date"]).days if w["join_date"] else 0
        jobs_completed = w["jobs_completed"]
        is_cold_start = (tenure_days < 21) or (jobs_completed < 3)
        jobs_offered = w["jobs_offered_count"]
        cancellation_rate = (w["cancellation_count"] / jobs_offered) if jobs_offered > 0 else 0.0

        # has_required_skill is set by get_candidate_workers: True when no
        # skill was requested, or when the worker's skills array actually
        # contains it; False only in the fallback-widened case.
        skill_match = int(w.get("has_required_skill", True))

        rows.append({
            "worker_id": w["worker_id"],
            "name": w["name"],
            "distance_km": float(w["distance_km"]),
            "budget_pkr": job.budget_pkr,
            "worker_hourly_rate_pkr": w["hourly_rate_pkr"],
            "worker_rating_smoothed": smoothed_rating(w["avg_rating"], w["num_reviews"]),
            "worker_num_reviews": w["num_reviews"],
            "worker_jobs_completed": jobs_completed,
            "worker_tenure_days": tenure_days,
            "worker_is_verified": int(w["is_verified"]),
            "worker_response_time_min": float(w["avg_response_time_min"] or DEFAULT_RESPONSE_TIME_MIN),
            "worker_cancellation_rate": cancellation_rate,
            "urgency": URGENCY_MAP.get(job.urgency, 1),
            "skill_match": skill_match,
            "is_cold_start": is_cold_start,
        })
    return pd.DataFrame(rows)


def rank_workers(df: pd.DataFrame, top_n: int, epsilon: float = 0.08) -> pd.DataFrame:
    df = df.copy()
    df["model_raw_score"] = model.predict(df[FEATURES])
    df["rule_raw_score"] = df.apply(rule_based_score, axis=1)

    # Rank-fusion blend: model score and rule score live on different, not
    # directly comparable scales (LambdaRank's raw output isn't a probability),
    # so blend by NORMALIZED RANK within this candidate set rather than by
    # raw value. 1.0 = top of that ranking, 0.0 = bottom.
    n = len(df)
    if n > 1:
        model_pct = df["model_raw_score"].rank(pct=True)
        rule_pct = df["rule_raw_score"].rank(pct=True)
    else:
        model_pct = pd.Series([1.0], index=df.index)
        rule_pct = pd.Series([1.0], index=df.index)

    df["blended_score"] = MODEL_TRUST_WEIGHT * model_pct + (1 - MODEL_TRUST_WEIGHT) * rule_pct

    # Exploration bonus for cold-start workers, applied AFTER blending so it
    # can still occasionally lift a new worker above a well-established one.
    is_cold = df["is_cold_start"].astype(bool)
    boost = np.where(is_cold, np.random.uniform(0, epsilon, size=len(df)), 0.0)
    df["final_score"] = df["blended_score"] + boost

    return df.sort_values("final_score", ascending=False).head(top_n)


@app.post("/jobs", response_model=JobResponse)
def create_job_and_recommend(job: JobRequest):
    if job.category not in VALID_CATEGORIES:
        raise HTTPException(400, f"Unknown category: {job.category}")

    job_id = f"J-{uuid.uuid4().hex[:8]}"
    conn = get_conn()
    try:
        # 1. persist the job FIRST -- match_events has a FK against jobs.job_id
        create_job(conn, job_id, job.customer_id, job.category, job.lat, job.lon,
                   job.urgency, job.budget_pkr, job.required_skill)

        # 2. candidate generation -- queried live from Postgres/PostGIS.
        # Tries a strict skill match first, falls back to the whole category
        # if too few skilled candidates exist nearby (see db.get_candidate_workers).
        candidates = get_candidate_workers(conn, job.category, job.lat, job.lon,
                                            job.radius_km, job.required_skill)
        if not candidates:
            return JobResponse(job_id=job_id, recommendations=[])

        # 3. feature assembly + 4. ranking + exploration boost
        features_df = build_feature_frame(candidates, job)
        ranked = rank_workers(features_df, job.top_n)

        # 5. log every SHOWN candidate -- needed as negatives for future training
        for pos, (_, row) in enumerate(ranked.iterrows(), start=1):
            log_shown(
                conn, job_id=job_id, worker_id=row["worker_id"],
                rank_position=pos, model_score=float(row["final_score"]),
                distance_km=float(row["distance_km"]),
                rating_smoothed=float(row["worker_rating_smoothed"]),
                jobs_completed=int(row["worker_jobs_completed"]),
                tenure_days=int(row["worker_tenure_days"]),
                skill_match=bool(row["skill_match"]),
            )
    finally:
        conn.close()

    recommendations = [
        RankedWorker(
            worker_id=row["worker_id"], name=row["name"],
            score=round(float(row["final_score"]), 4),
            distance_km=round(float(row["distance_km"]), 2),
            rating_smoothed=round(float(row["worker_rating_smoothed"]), 2),
            is_cold_start=bool(row["is_cold_start"]),
            has_required_skill=bool(row["skill_match"]),
        )
        for _, row in ranked.iterrows()
    ]
    return JobResponse(job_id=job_id, recommendations=recommendations)


@app.post("/feedback")
def record_feedback(fb: FeedbackRequest):
    conn = get_conn()
    try:
        if fb.event_type == "hired":
            log_hired(conn, fb.job_id, fb.worker_id)
        elif fb.event_type == "completed":
            log_completed(conn, fb.job_id, fb.worker_id)
        elif fb.event_type == "cancelled":
            log_cancelled(conn, fb.job_id, fb.worker_id)
        elif fb.event_type == "rated":
            if fb.rating is None:
                raise HTTPException(400, "rating is required when event_type='rated'")
            log_rating(conn, fb.job_id, fb.worker_id, fb.rating)
    finally:
        conn.close()
    return {"status": "recorded"}


@app.get("/health")
def health():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM workers;")
            n_workers = cur.fetchone()[0]
    finally:
        conn.close()
    return {
        "status": "ok",
        "workers_in_db": n_workers,
        "active_model": active_model_path.name,
        "using_real_model": USING_REAL_MODEL,
        "model_trust_weight": MODEL_TRUST_WEIGHT,
    }