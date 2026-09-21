"""
Synthetic data generator for a Pakistani blue-collar services marketplace
(plumbing, carpentry, electrician, mechanic, masonry) worker-job matching model.

Outputs three CSVs:
  1. workers.csv        - worker profile/features
  2. jobs.csv            - job postings
  3. interactions.csv    - historical (job, worker, outcome) pairs for training a ranker
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

rng = np.random.default_rng(42)
random.seed(42)

# ---------------------------------------------------------------------------
# 1. Reference data: Pakistani cities with approximate lat/lon bounding boxes
# ---------------------------------------------------------------------------
CITIES = {
    "Karachi":     {"lat": 24.8607, "lon": 67.0011, "spread": 0.18, "weight": 0.26},
    "Lahore":      {"lat": 31.5497, "lon": 74.3436, "spread": 0.15, "weight": 0.22},
    "Islamabad":   {"lat": 33.6844, "lon": 73.0479, "spread": 0.10, "weight": 0.10},
    "Rawalpindi":  {"lat": 33.5651, "lon": 73.0169, "spread": 0.08, "weight": 0.08},
    "Faisalabad":  {"lat": 31.4180, "lon": 73.0790, "spread": 0.08, "weight": 0.08},
    "Multan":      {"lat": 30.1575, "lon": 71.5249, "spread": 0.07, "weight": 0.06},
    "Peshawar":    {"lat": 34.0151, "lon": 71.5249, "spread": 0.07, "weight": 0.06},
    "Hyderabad":   {"lat": 25.3960, "lon": 68.3578, "spread": 0.06, "weight": 0.05},
    "Gujranwala":  {"lat": 32.1877, "lon": 74.1945, "spread": 0.06, "weight": 0.05},
    "Quetta":      {"lat": 30.1798, "lon": 66.9750, "spread": 0.07, "weight": 0.04},
}
city_names = list(CITIES.keys())
city_weights = np.array([CITIES[c]["weight"] for c in city_names])
city_weights = city_weights / city_weights.sum()

CATEGORIES = ["Plumbing", "Carpentry", "Electrician", "Mechanic", "Masonry"]

SKILLS_BY_CATEGORY = {
    "Plumbing":    ["pipe fitting", "leak repair", "water tank install", "drainage", "bathroom fitting", "motor repair"],
    "Carpentry":   ["furniture repair", "door/window fitting", "false ceiling", "wardrobe making", "polishing", "custom furniture"],
    "Electrician": ["wiring", "fan/light install", "MCB/breaker", "inverter/UPS install", "solar panel wiring", "AC electrical"],
    "Mechanic":    ["engine repair", "AC repair", "denting/painting", "electrical (auto)", "bike repair", "oil change"],
    "Masonry":     ["brickwork", "plastering", "tile fitting", "waterproofing", "boundary wall", "flooring"],
}

FIRST_NAMES_M = ["Muhammad", "Ahmed", "Ali", "Usman", "Bilal", "Imran", "Waseem", "Tariq", "Shahid", "Naveed",
                  "Zeeshan", "Kashif", "Faisal", "Adnan", "Rashid", "Javed", "Aftab", "Irfan", "Sajid", "Yasir"]
LAST_NAMES = ["Khan", "Ahmed", "Hussain", "Malik", "Butt", "Sheikh", "Qureshi", "Baloch", "Rajput", "Awan",
              "Chaudhry", "Abbasi", "Soomro", "Memon", "Ansari", "Gill", "Warraich", "Bhatti", "Niazi", "Raza"]

N_WORKERS = 600
N_JOBS = 2500
PLATFORM_LAUNCH = datetime(2023, 1, 1)
TODAY = datetime(2026, 7, 14)

def sample_city():
    return np.random.choice(city_names, p=city_weights)

def jitter_latlon(city):
    c = CITIES[city]
    lat = c["lat"] + rng.normal(0, c["spread"] / 3)
    lon = c["lon"] + rng.normal(0, c["spread"] / 3)
    return round(lat, 5), round(lon, 5)

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(a))

def random_join_date():
    days_range = (TODAY - PLATFORM_LAUNCH).days
    d = rng.integers(0, days_range)
    return PLATFORM_LAUNCH + timedelta(days=int(d))

# ---------------------------------------------------------------------------
# 2. Workers
# ---------------------------------------------------------------------------
workers = []
for i in range(1, N_WORKERS + 1):
    city = sample_city()
    lat, lon = jitter_latlon(city)
    category = np.random.choice(CATEGORIES)
    n_skills = rng.integers(1, 4)
    skills = list(np.random.choice(SKILLS_BY_CATEGORY[category], size=min(n_skills, len(SKILLS_BY_CATEGORY[category])), replace=False))

    join_date = random_join_date()
    tenure_days = (TODAY - join_date).days

    # jobs_completed correlated with tenure, with noise; new workers (low tenure) -> low/zero jobs
    expected_jobs = max(0, tenure_days / 15.0)  # roughly a job every ~2 weeks when active
    jobs_completed = int(max(0, rng.poisson(lam=max(expected_jobs * rng.uniform(0.2, 1.0), 0.01))))
    jobs_completed = min(jobs_completed, 400)

    # is this a "cold start" worker? tenure < 21 days or jobs_completed < 3
    is_new = (tenure_days < 21) or (jobs_completed < 3)

    num_reviews = int(max(0, jobs_completed * rng.uniform(0.4, 0.85)))  # not every job gets reviewed

    if num_reviews == 0:
        avg_rating = np.nan  # no rating yet -> true cold start
    else:
        # underlying "true skill" quality per worker
        true_quality = np.clip(rng.normal(4.3, 0.45), 2.5, 5.0)
        avg_rating = round(float(np.clip(rng.normal(true_quality, 0.3), 1.0, 5.0)), 2)

    age_years = rng.integers(20, 58)
    hourly_rate = int(rng.integers(400, 2500))  # PKR/hour, varies by category & city cost of living
    if city in ("Karachi", "Lahore", "Islamabad"):
        hourly_rate = int(hourly_rate * 1.15)

    response_time_min = int(np.clip(rng.exponential(20) + 5, 3, 180))
    verified = bool(rng.random() < (0.85 if tenure_days > 60 else 0.4))
    has_own_tools = bool(rng.random() < 0.8)
    cancellation_rate = round(float(np.clip(rng.beta(1.5, 20), 0, 0.5)), 3)

    workers.append({
        "worker_id": f"W{i:05d}",
        "name": f"{np.random.choice(FIRST_NAMES_M)} {np.random.choice(LAST_NAMES)}",
        "city": city,
        "lat": lat,
        "lon": lon,
        "category": category,
        "skills": ";".join(skills),
        "age": int(age_years),
        "join_date": join_date.date().isoformat(),
        "tenure_days": int(tenure_days),
        "jobs_completed": jobs_completed,
        "num_reviews": num_reviews,
        "avg_rating": avg_rating,
        "hourly_rate_pkr": hourly_rate,
        "response_time_min": response_time_min,
        "is_verified": verified,
        "has_own_tools": has_own_tools,
        "cancellation_rate": cancellation_rate,
        "is_cold_start": is_new,
    })

workers_df = pd.DataFrame(workers)

# ---------------------------------------------------------------------------
# 3. Jobs
# ---------------------------------------------------------------------------
URGENCY = ["low", "medium", "high", "emergency"]
urgency_weights = [0.30, 0.40, 0.20, 0.10]

jobs = []
for j in range(1, N_JOBS + 1):
    city = sample_city()
    lat, lon = jitter_latlon(city)
    category = np.random.choice(CATEGORIES)
    created = PLATFORM_LAUNCH + timedelta(days=int(rng.integers(0, (TODAY - PLATFORM_LAUNCH).days)))
    urgency = np.random.choice(URGENCY, p=urgency_weights)
    base_budget = {
        "Plumbing": 1500, "Carpentry": 3000, "Electrician": 1800, "Mechanic": 2500, "Masonry": 5000
    }[category]
    budget = int(np.clip(rng.normal(base_budget, base_budget * 0.35), base_budget * 0.3, base_budget * 3))

    jobs.append({
        "job_id": f"J{j:06d}",
        "city": city,
        "lat": lat,
        "lon": lon,
        "category": category,
        "urgency": urgency,
        "budget_pkr": budget,
        "created_at": created.isoformat(),
    })

jobs_df = pd.DataFrame(jobs)

# ---------------------------------------------------------------------------
# 4. Interactions (job x candidate worker -> shown / hired label)
#    For each job, sample ~8-15 candidate workers from the SAME category
#    (mix of near + far, mix of experienced + cold-start) to mimic what a
#    candidate-generation stage would retrieve. Then simulate a hire decision.
# ---------------------------------------------------------------------------
def smoothed_rating(avg_rating, num_reviews, prior=4.2, prior_weight=5):
    if pd.isna(avg_rating) or num_reviews == 0:
        return prior  # no evidence -> platform prior
    return (prior * prior_weight + avg_rating * num_reviews) / (prior_weight + num_reviews)

interactions = []
workers_by_cat = {c: workers_df[workers_df.category == c] for c in CATEGORIES}

for _, job in jobs_df.iterrows():
    pool = workers_by_cat[job.category]
    if len(pool) < 3:
        continue
    n_candidates = min(len(pool), rng.integers(8, 16))
    candidates = pool.sample(n=n_candidates, random_state=int(rng.integers(0, 1e9)))

    scores = []
    for _, w in candidates.iterrows():
        dist_km = round(haversine_km(job.lat, job.lon, w.lat, w.lon), 2)
        s_rating = smoothed_rating(w.avg_rating, w.num_reviews)

        # ground-truth utility function used ONLY to simulate realistic hire behavior
        utility = (
            - 0.12 * dist_km
            + 1.4 * (s_rating - 3.0)
            + 0.015 * np.log1p(w.jobs_completed)
            + 0.6 * float(w.is_verified)
            - 0.05 * (w.response_time_min / 10)
            - 3.0 * w.cancellation_rate
            + rng.normal(0, 0.8)  # noise / unobserved factors
        )
        # small exploration boost simulated for cold-start workers actually being SHOWN at all
        scores.append((w.worker_id, dist_km, s_rating, utility))

    # convert utility -> hire probability via softmax over candidates (one hire per job, usually)
    utils = np.array([s[3] for s in scores])
    probs = np.exp(utils - utils.max())
    probs = probs / probs.sum()
    hired_idx = np.random.choice(len(scores), p=probs)

    for idx, (worker_id, dist_km, s_rating, utility) in enumerate(scores):
        w = candidates.iloc[idx]
        interactions.append({
            "job_id": job.job_id,
            "worker_id": worker_id,
            "category": job.category,
            "distance_km": dist_km,
            "category_match": 1,  # candidate gen already filters by category
            "urgency": job.urgency,
            "budget_pkr": job.budget_pkr,
            "worker_hourly_rate_pkr": int(w.hourly_rate_pkr),
            "worker_rating_smoothed": round(s_rating, 3),
            "worker_num_reviews": int(w.num_reviews),
            "worker_jobs_completed": int(w.jobs_completed),
            "worker_tenure_days": int(w.tenure_days),
            "worker_is_verified": bool(w.is_verified),
            "worker_response_time_min": int(w.response_time_min),
            "worker_cancellation_rate": float(w.cancellation_rate),
            "worker_is_cold_start": bool(w.is_cold_start),
            "was_hired": int(idx == hired_idx),   # label for learning-to-rank / classification
        })

interactions_df = pd.DataFrame(interactions)

# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
workers_df.to_csv("data/synthetic/workers.csv", index=False)
jobs_df.to_csv("data/synthetic/jobs.csv", index=False)
interactions_df.to_csv("data/synthetic/interactions.csv", index=False)

print("workers:", workers_df.shape)
print("jobs:", jobs_df.shape)
print("interactions:", interactions_df.shape)
print("\ncold-start worker %:", round(workers_df.is_cold_start.mean() * 100, 1))
print("\nworkers sample:\n", workers_df.head(3).to_string())
print("\ninteractions sample:\n", interactions_df.head(5).to_string())
