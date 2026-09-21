-- ============================================================
-- Worker Matching Platform - Core Schema (Postgres + PostGIS)
-- ============================================================
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE workers (
    worker_id           TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    city                TEXT NOT NULL,
    location             GEOGRAPHY(POINT, 4326) NOT NULL,  -- lat/lon, indexed for geo queries
    category            TEXT NOT NULL,
    skills              TEXT[],
    age                 INT,
    join_date           DATE NOT NULL,
    hourly_rate_pkr     INT,
    is_verified         BOOLEAN DEFAULT FALSE,
    has_own_tools       BOOLEAN DEFAULT TRUE,
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_workers_location ON workers USING GIST (location);
CREATE INDEX idx_workers_category ON workers (category);

-- Aggregated/derived stats -- updated incrementally, NOT recomputed from
-- scratch on every request. Keep this separate from the raw workers table
-- so writes to it (frequent) don't lock the profile table (rare writes).
CREATE TABLE worker_stats (
    worker_id            TEXT PRIMARY KEY REFERENCES workers(worker_id),
    jobs_completed       INT DEFAULT 0,
    num_reviews          INT DEFAULT 0,
    avg_rating           NUMERIC(3,2),
    rating_smoothed       NUMERIC(4,3),          -- recomputed whenever a review lands
    cancellation_count    INT DEFAULT 0,
    jobs_offered_count    INT DEFAULT 0,          -- denominator for cancellation_rate
    avg_response_time_min NUMERIC(6,2),
    last_active_at        TIMESTAMPTZ,
    updated_at            TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE jobs (
    job_id              TEXT PRIMARY KEY,
    customer_id          TEXT NOT NULL,
    category             TEXT NOT NULL,
    location              GEOGRAPHY(POINT, 4326) NOT NULL,
    urgency               TEXT CHECK (urgency IN ('low','medium','high','emergency')),
    budget_pkr            INT,
    status                TEXT DEFAULT 'open',    -- open | matched | completed | cancelled
    created_at            TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_jobs_location ON jobs USING GIST (location);

-- ============================================================
-- Event log -- THIS is your future training data. Log everything,
-- including negatives (shown-but-not-hired). Append-only, never update.
-- ============================================================
CREATE TABLE match_events (
    event_id             BIGSERIAL PRIMARY KEY,
    job_id                TEXT NOT NULL REFERENCES jobs(job_id),
    worker_id             TEXT NOT NULL REFERENCES workers(worker_id),
    event_type            TEXT NOT NULL CHECK (event_type IN
                             ('shown','hired','not_hired','completed','cancelled','rated')),
    -- snapshot the features AS THEY WERE AT MATCH TIME -- critical, because
    -- worker_stats changes over time and you must not leak future info
    -- into a past training row.
    distance_km            NUMERIC(7,2),
    rank_position          INT,               -- where this worker was ranked when shown
    model_score            NUMERIC(8,5),
    rating_smoothed_at_time NUMERIC(4,3),
    jobs_completed_at_time  INT,
    tenure_days_at_time      INT,
    post_job_rating          NUMERIC(3,2),      -- filled later when event_type='rated'
    model_version             TEXT,             -- which model produced this ranking
    created_at                TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_events_job ON match_events (job_id);
CREATE INDEX idx_events_worker ON match_events (worker_id);
CREATE INDEX idx_events_created ON match_events (created_at);

-- Model registry -- track every trained model and its offline eval score
CREATE TABLE model_registry (
    model_version         TEXT PRIMARY KEY,
    trained_at             TIMESTAMPTZ DEFAULT now(),
    training_data_start    TIMESTAMPTZ,
    training_data_end      TIMESTAMPTZ,
    n_training_rows        INT,
    ndcg_at_3               NUMERIC(5,4),
    ndcg_at_5               NUMERIC(5,4),
    is_production           BOOLEAN DEFAULT FALSE,
    artifact_path            TEXT               -- e.g. s3://bucket/models/v12.txt
);
