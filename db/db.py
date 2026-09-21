"""
db.py -- Postgres access layer. Workers/jobs are read from and written to
Postgres directly -- this is the single source of truth the live service
queries against (not a CSV snapshot), so foreign keys always resolve.
"""

import psycopg2
import psycopg2.extras
import os
from dotenv import load_dotenv
load_dotenv()
DB_DSN = os.getenv("DATABASE_URL")


def get_conn():
    return psycopg2.connect(DB_DSN)


def create_job(conn, job_id: str, customer_id: str, category: str, lat: float, lon: float,
               urgency: str, budget_pkr: int, required_skill: str = None):
    """Must be called BEFORE logging any match_events for this job (FK constraint).
    required_skill is the subcategory the job needs, e.g. 'engine repair' or
    'denting/painting' for a Mechanic job. Optional -- categories without
    meaningful subcategories can leave it NULL."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO jobs (job_id, customer_id, category, location, urgency, budget_pkr, required_skill)
            VALUES (%s,%s,%s, ST_SetSRID(ST_MakePoint(%s,%s),4326)::geography, %s,%s,%s)
            ON CONFLICT (job_id) DO NOTHING;
        """, (job_id, customer_id, category, lon, lat, urgency, budget_pkr, required_skill))
    conn.commit()


_CANDIDATE_QUERY = """
    SELECT
        w.worker_id, w.name, w.city, w.hourly_rate_pkr, w.is_verified, w.join_date,
        w.skills,
        ST_Distance(
            w.location,
            ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326)::geography
        ) / 1000.0 AS distance_km,
        COALESCE(ws.jobs_completed, 0)      AS jobs_completed,
        COALESCE(ws.num_reviews, 0)         AS num_reviews,
        ws.avg_rating,
        COALESCE(ws.cancellation_count, 0)   AS cancellation_count,
        COALESCE(ws.jobs_offered_count, 0)   AS jobs_offered_count,
        ws.avg_response_time_min
    FROM workers w
    LEFT JOIN worker_stats ws ON w.worker_id = ws.worker_id
    WHERE w.category = %(category)s
      AND ST_DWithin(
            w.location,
            ST_SetSRID(ST_MakePoint(%(lon)s, %(lat)s), 4326)::geography,
            %(radius_m)s
          )
      {skill_clause}
    ORDER BY distance_km
    LIMIT %(limit)s;
"""

MIN_SKILLED_CANDIDATES = 5  # below this, widen the search rather than show too few options


def get_candidate_workers(conn, category: str, lat: float, lon: float, radius_km: float,
                           required_skill: str = None, limit: int = 50):
    """Candidate generation stage -- geo + category (+ skill) filter, done in
    Postgres/PostGIS rather than in-process pandas, so it scales with the real
    worker table and always reflects the current roster (including workers
    added via ETL).

    When required_skill is given, this first tries a STRICT filter (worker's
    skills array must contain it) -- e.g. an 'engine repair' job should not
    even see denting/painting-only mechanics. If that strict pass comes back
    with too few candidates (new/thin market), it falls back to the full
    category roster so the customer still gets matches, and flags each row's
    has_required_skill so ranking can penalize the mismatch instead of hiding
    it entirely.
    """
    params = {"category": category, "lat": lat, "lon": lon,
              "radius_m": radius_km * 1000, "limit": limit}

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if required_skill:
            params["skill"] = required_skill
            strict_query = _CANDIDATE_QUERY.format(skill_clause="AND %(skill)s = ANY(w.skills)")
            cur.execute(strict_query, params)
            rows = cur.fetchall()
            for r in rows:
                r["has_required_skill"] = True
            if len(rows) >= MIN_SKILLED_CANDIDATES:
                return rows

            # Fallback: widen to the whole category, mark who actually has the skill.
            fallback_query = _CANDIDATE_QUERY.format(skill_clause="")
            cur.execute(fallback_query, params)
            all_rows = cur.fetchall()
            skilled_ids = {r["worker_id"] for r in rows}
            for r in all_rows:
                r["has_required_skill"] = r["worker_id"] in skilled_ids
            return all_rows

        query = _CANDIDATE_QUERY.format(skill_clause="")
        cur.execute(query, params)
        rows = cur.fetchall()
        for r in rows:
            r["has_required_skill"] = True  # no skill was requested, so nothing to mismatch on
        return rows


def log_shown(conn, job_id: str, worker_id: str, rank_position: int, model_score: float,
              distance_km: float, rating_smoothed: float, jobs_completed: int,
              tenure_days: int, skill_match: bool = True, model_version: str = "rule_based_v1"):
    """Call once per candidate returned in the ranked list. Also increments
    jobs_offered_count -- the denominator behind cancellation_rate."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO match_events
                (job_id, worker_id, event_type, distance_km, rank_position, model_score,
                 rating_smoothed_at_time, jobs_completed_at_time, tenure_days_at_time,
                 skill_match_at_time, model_version)
            VALUES (%s,%s,'shown',%s,%s,%s,%s,%s,%s,%s,%s);
        """, (job_id, worker_id, distance_km, rank_position, model_score,
              rating_smoothed, jobs_completed, tenure_days, skill_match, model_version))
        cur.execute("""
            INSERT INTO worker_stats (worker_id, jobs_offered_count)
            VALUES (%s, 1)
            ON CONFLICT (worker_id) DO UPDATE
                SET jobs_offered_count = worker_stats.jobs_offered_count + 1;
        """, (worker_id,))
    conn.commit()


def log_hired(conn, job_id: str, worker_id: str):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO match_events (job_id, worker_id, event_type)
            VALUES (%s,%s,'hired');
        """, (job_id, worker_id))
    conn.commit()


def log_completed(conn, job_id: str, worker_id: str):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO match_events (job_id, worker_id, event_type)
            VALUES (%s,%s,'completed');
        """, (job_id, worker_id))
        cur.execute("""
            UPDATE worker_stats
            SET jobs_completed = jobs_completed + 1, last_active_at = now(), updated_at = now()
            WHERE worker_id = %s;
        """, (worker_id,))
    conn.commit()


def log_cancelled(conn, job_id: str, worker_id: str):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO match_events (job_id, worker_id, event_type)
            VALUES (%s,%s,'cancelled');
        """, (job_id, worker_id))
        cur.execute("""
            UPDATE worker_stats
            SET cancellation_count = cancellation_count + 1, updated_at = now()
            WHERE worker_id = %s;
        """, (worker_id,))
    conn.commit()


def log_rating(conn, job_id: str, worker_id: str, rating: float):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO match_events (job_id, worker_id, event_type, post_job_rating)
            VALUES (%s,%s,'rated',%s);
        """, (job_id, worker_id, rating))
        cur.execute("""
            UPDATE worker_stats
            SET num_reviews = num_reviews + 1,
                avg_rating = (COALESCE(avg_rating,0) * num_reviews + %s) / (num_reviews + 1),
                rating_smoothed = (4.2 * 5 + (COALESCE(avg_rating,0) * num_reviews + %s)) / (5 + num_reviews + 1),
                updated_at = now()
            WHERE worker_id = %s;
        """, (rating, rating, worker_id))
    conn.commit()