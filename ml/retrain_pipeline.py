"""
Retraining pipeline -- triggered by cron (see README for the crontab line).

    0 2 * * *  /usr/bin/python3 /opt/app/ml/retrain_pipeline.py >> logs/retrain.log 2>&1

1. Pulls match_events from Postgres for the lookback window
2. Archives a versioned Parquet snapshot to data/training-snapshots/
3. Retrains LambdaRank with a TIME-based (not random) split
4. Evaluates NDCG@3/@5 on the held-out recent window
5. Compares against current production model in model_registry
6. Promotes only if it clears MIN_IMPROVEMENT; always logs the attempt
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
from pathlib import Path
import psycopg2
import shutil
import uuid
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db.db import get_conn  # reuse the same connection helper as the live service

BASE_DIR = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = BASE_DIR / "data" / "training-snapshots"
MODEL_DIR = BASE_DIR / "ml" / "models"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MIN_ROWS_TO_TRAIN = 500
MIN_IMPROVEMENT = 0.005
LOOKBACK_DAYS = 90

FEATURES = [
    "distance_km", "budget_pkr", "worker_hourly_rate_pkr",
    "worker_rating_smoothed", "worker_num_reviews", "worker_jobs_completed",
    "worker_tenure_days", "worker_is_verified", "worker_response_time_min",
    "worker_cancellation_rate", "urgency", "skill_match",
]


def pull_training_data(conn, since: datetime) -> pd.DataFrame:
    query = """
        SELECT
            job_id, worker_id,
            distance_km,
            rating_smoothed_at_time  AS worker_rating_smoothed,
            jobs_completed_at_time   AS worker_jobs_completed,
            tenure_days_at_time      AS worker_tenure_days,
            skill_match_at_time      AS skill_match,
            model_score,
            CASE WHEN event_type = 'hired' THEN 1 ELSE 0 END AS was_hired,
            created_at
        FROM match_events
        WHERE event_type IN ('shown', 'hired')
          AND created_at >= %s
        ORDER BY job_id;
    """
    return pd.read_sql(query, conn, params=(since,))


def evaluate_ndcg(model, df, features, k=3):
    df = df.copy()
    df["pred"] = model.predict(df[features])
    scores = []
    for _, group in df.groupby("job_id"):
        group = group.sort_values("pred", ascending=False)
        rels = group["was_hired"].values[:k]
        dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rels))
        ideal = sorted(group["was_hired"].values, reverse=True)[:k]
        idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
        if idcg > 0:
            scores.append(dcg / idcg)
    return float(np.mean(scores)) if scores else 0.0


def get_current_production(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT model_version, ndcg_at_3, artifact_path FROM model_registry WHERE is_production = TRUE LIMIT 1;")
        return cur.fetchone()


def register_model(conn, version, start, end, n_rows, ndcg3, ndcg5, path, promote):
    with conn.cursor() as cur:
        if promote:
            cur.execute("UPDATE model_registry SET is_production = FALSE WHERE is_production = TRUE;")
        cur.execute("""
            INSERT INTO model_registry
                (model_version, training_data_start, training_data_end, n_training_rows,
                 ndcg_at_3, ndcg_at_5, is_production, artifact_path)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s);
        """, (version, start, end, n_rows, ndcg3, ndcg5, promote, str(path)))
    conn.commit()


def main():
    conn = get_conn()
    since = datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)
    end = datetime.utcnow()

    df = pull_training_data(conn, since)
    print(f"[retrain] pulled {len(df)} rows since {since.isoformat()}")

    if len(df) < MIN_ROWS_TO_TRAIN:
        print(f"[retrain] not enough data yet ({len(df)} < {MIN_ROWS_TO_TRAIN}), skipping.")
        conn.close()
        return

    snapshot_path = SNAPSHOT_DIR / f"events_{end.date().isoformat()}.parquet"
    df.to_parquet(snapshot_path)
    print(f"[retrain] snapshot saved: {snapshot_path}")

    df = df.sort_values("created_at")
    cutoff_idx = int(len(df) * 0.8)
    cutoff_time = df.iloc[cutoff_idx]["created_at"]
    train_df = df[df["created_at"] < cutoff_time]
    valid_df = df[df["created_at"] >= cutoff_time]

    train_groups = train_df.groupby("job_id", sort=False).size().values
    train_set = lgb.Dataset(train_df[FEATURES], label=train_df["was_hired"], group=train_groups)

    params = {
        "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1, 3, 5],
        "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 20, "verbose": -1,
    }
    new_model = lgb.train(params, train_set, num_boost_round=300)

    new_ndcg3 = evaluate_ndcg(new_model, valid_df, FEATURES, k=3)
    new_ndcg5 = evaluate_ndcg(new_model, valid_df, FEATURES, k=5)
    print(f"[retrain] new model NDCG@3={new_ndcg3:.4f}  NDCG@5={new_ndcg5:.4f}")

    current = get_current_production(conn)
    version = f"v_{end.strftime('%Y%m%d_%H%M')}_{uuid.uuid4().hex[:6]}"
    model_path = MODEL_DIR / f"{version}.txt"
    new_model.save_model(str(model_path))

    promote = False
    if current is None:
        print("[retrain] no production model yet -> promoting first model.")
        promote = True
    else:
        prod_version, prod_ndcg3, prod_path = current
        print(f"[retrain] current production: {prod_version} (NDCG@3={prod_ndcg3:.4f})")
        if new_ndcg3 - float(prod_ndcg3) >= MIN_IMPROVEMENT:
            promote = True
            print("[retrain] new model beats production -> promoting.")
        else:
            print("[retrain] new model does NOT clear the improvement bar -> keeping current production model.")

    register_model(conn, version, since, end, len(train_df), new_ndcg3, new_ndcg5, model_path, promote)

    if promote:
        live_path = MODEL_DIR / "production.txt"
        shutil.copyfile(model_path, live_path)
        print(f"[retrain] promoted {version} to production ({live_path}).")

    conn.close()


if __name__ == "__main__":
    main()