"""
One-off/manual training script -- trains on the interactions.csv (synthetic
data, or a hand-built real snapshot). For ONGOING production retraining
against live Postgres data, use retrain_pipeline.py instead -- that one
has time-based splitting, an evaluation gate, and model registry logging.
Use this script for initial experimentation only.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "synthetic" / "interactions.csv"
MODEL_OUT = BASE_DIR / "ml" / "models" / "worker_ranker.txt"

df = pd.read_csv(DATA_PATH)
df["worker_is_verified"] = df["worker_is_verified"].astype(int)
df["worker_is_cold_start"] = df["worker_is_cold_start"].astype(int)
df["skill_match"] = df["skill_match"].astype(int)
df["urgency"] = df["urgency"].astype("category").cat.codes

FEATURES = [
    "distance_km", "budget_pkr", "worker_hourly_rate_pkr",
    "worker_rating_smoothed", "worker_num_reviews", "worker_jobs_completed",
    "worker_tenure_days", "worker_is_verified", "worker_response_time_min",
    "worker_cancellation_rate", "urgency", "skill_match",
]
LABEL = "was_hired"
GROUP_KEY = "job_id"

splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(splitter.split(df, groups=df[GROUP_KEY]))
train_df = df.iloc[train_idx].sort_values(GROUP_KEY)
test_df = df.iloc[test_idx].sort_values(GROUP_KEY)

train_group_sizes = train_df.groupby(GROUP_KEY, sort=False).size().values
test_group_sizes = test_df.groupby(GROUP_KEY, sort=False).size().values

train_set = lgb.Dataset(train_df[FEATURES], label=train_df[LABEL], group=train_group_sizes)
test_set = lgb.Dataset(test_df[FEATURES], label=test_df[LABEL], group=test_group_sizes, reference=train_set)

params = {
    "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1, 3, 5],
    "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 20, "verbose": -1,
}

model = lgb.train(
    params, train_set, num_boost_round=300, valid_sets=[test_set],
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
)

print("\nFeature importance:")
for feat, imp in sorted(zip(FEATURES, model.feature_importance()), key=lambda x: -x[1]):
    print(f"  {feat:30s} {imp}")

MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
model.save_model(str(MODEL_OUT))
print(f"\nModel saved to {MODEL_OUT}")