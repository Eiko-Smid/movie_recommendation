import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv 

from math import ceil

from fastapi import HTTPException, status
import pandas as pd


# Database settings
MV_NAME = "eligible_users"          # name of materialized view
MIN_N_USER_RATINGS = 6              # users must have >=MIN_N_USER_RATINGS ratings to be part of MV
ASSUMED_AVG_PER_USER = 50.0         # rough avg ratings per eligible user (for sizing K)
RANDOM_OVERSAMPLE = 1.10            # oversample users by ~10% to hit target size
TARGET_FLOOR_RATIO = 0.9            # accept result if >=90% of n_rows
UNDERSHOOT_RETRY_FACTOR = 1.5       # if we undershoot badly, bump K by 50% once
AUTO_REFRESH_MV = False             # keep False to stay within +20% runtime budget


def _get_engine() -> Engine:
    load_dotenv()
    DB_URL = os.getenv("DB_URL")
    if not DB_URL:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection URL not found in environment variables."
        )
    return create_engine(DB_URL)


def _mv_exists(engine: Engine) -> bool:
    q = text("""
        SELECT 1
        FROM pg_matviews
        WHERE schemaname = current_schema()
          AND matviewname = :mv_name
        LIMIT 1;
    """)
    with engine.connect() as conn:
        res = conn.execute(q, {"mv_name": MV_NAME}).first()
    return res is not None


def _ensure_indexes(engine: Engine) -> None:
    # Create helpful indexes (idempotent)
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{MV_NAME}_userid
            ON {MV_NAME}("userId");
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_ratings_userid
            ON ratings("userId");
        """))


def _create_mv_if_missing(engine: Engine) -> None:
    if _mv_exists(engine):
        return
    # Create MV (run once; safe to call every time)
    # Use a TX for CREATE; REFRESH CONCURRENTLY can't be in a TX, so we do not refresh here.
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE MATERIALIZED VIEW IF NOT EXISTS {MV_NAME} AS
            SELECT "userId"
            FROM ratings
            GROUP BY "userId"
            HAVING COUNT(*) >= :min_r;
        """), {"min_r": MIN_N_USER_RATINGS})
    _ensure_indexes(engine)


def refresh_mv():
    '''
    Creates or refreshes the materialized view (MV) that stores all users ids
    who have rated 'MIN_N_USER_RATINGS' or more movies.

    Steps:
    1. Connects to the PostgreSQL database.
    2. Creates the materialized view if it does not exist yet.
    3. Ensures the necessary indexes exist (for fast refresh and lookups).
    4. Refreshes the MV so it contains the newest data from the 'ratings' table.
       - Uses a concurrent refresh if possible (no read lock).
       - Falls back to a normal refresh if concurrent mode is not supported.

    Parameters
    ----------

    Returns
    -------
    refreshed_concurrently: bool:
            True:   The MV was refreshed using CONCURRENTLY (non-blocking). 
                    Other code can still use the materialized view while the new one 
                    is getting builded
            False:  The MV was refreshed using the normal (blocking) method.
    '''
    # Get DB connection factory
    engine = _get_engine()

    # Run transaction and create mv if it does not exist 
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE MATERIALIZED VIEW IF NOT EXISTS {MV_NAME} AS
            SELECT "userId"
            FROM ratings
            GROUP BY "userId"
            HAVING COUNT(*) >= :min_r;
        """), {"min_r": MIN_N_USER_RATINGS})
        # Ensure unique index needed for CONCURRENTLY
        conn.execute(text(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{MV_NAME}_userid
            ON {MV_NAME}("userId");
        """))
        # Helpful base-table index (no-op if exists)
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_ratings_userid
            ON ratings("userId");
        """))

    # Refresh MV
    try:
        # Try a concurrent refresh of the MV -> new mv gets created while old one still 
        # exists -> mv can be accessed while updating.
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            # Optional: advisory lock so only one refresh runs at a time
            conn.execute(text("SELECT pg_try_advisory_lock(987654321);"))
            conn.execute(text(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {MV_NAME};"))
            conn.execute(text("SELECT pg_advisory_unlock(987654321);"))
        refreshed_concurrently = True
    except Exception:
        # Fallback if concurrent refresh isn’t possible
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(f"REFRESH MATERIALIZED VIEW {MV_NAME};"))
        refreshed_concurrently = False
    
    return refreshed_concurrently


def _estimate_avg_per_user(engine: Engine) -> float:
    """
    Estimate avg ratings per eligible user. Cheap and good enough.
    """
    q = text(f"""
        SELECT AVG(cnt)::float AS avg_cnt
        FROM (
          SELECT COUNT(*) AS cnt
          FROM ratings
          WHERE "userId" IN (SELECT "userId" FROM {MV_NAME})
          GROUP BY "userId"
        ) t;
    """)
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(q, conn)
        val = df.iloc[0]["avg_cnt"]
        return float(val) if val is not None else ASSUMED_AVG_PER_USER
    except Exception:
        return ASSUMED_AVG_PER_USER



def _load_full_histories_for_n_users(engine, n_users_target: int) -> pd.DataFrame:
    """
    Sample EXACTLY 'n_users_target' random eligible users (>=6 ratings) and return ALL of their ratings.
    """
    if n_users_target <= 0:
        raise ValueError("n_users_target must be > 0")

    # Check MV size (optional but helpful for a clear error)
    with engine.connect() as conn:
        mv_cnt = pd.read_sql_query(
            text('SELECT COUNT(*) AS c FROM eligible_users;'), conn
        ).iloc[0, 0]

    if mv_cnt < n_users_target:
        # Choose ONE of the following behaviors:

        # 1) Strict: complain clearly
        # raise ValueError(f"Requested {n_users_target} users, but MV has only {mv_cnt}. Refresh MV or lower target.")

        # 2) Lenient: use as many as available
        n_users_target = mv_cnt

    sql = text("""
        WITH sample_users AS (
          SELECT "userId"
          FROM eligible_users
          ORDER BY RANDOM()     -- different set every run
          LIMIT :k_users        -- EXACT user count
        )
        SELECT r."userId", r."movieId", r.rating, r."timestamp"
        FROM ratings r
        JOIN sample_users su USING ("userId");
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn, params={"k_users": int(n_users_target)})

    # Sanity log (optional)
    # print(f"[Loader] users_requested={n_users_target}, users_returned={df['userId'].nunique()}, rows={len(df)}")

    return df



def _sample_full_histories(engine: Engine, n_rows_target: int) -> pd.DataFrame:
    """
    Sample K random eligible users and fetch ALL their ratings.
    Targets ~n_rows_target rows overall (not exact, because full histories vary).
    """
    avg_cnt = _estimate_avg_per_user(engine)
    k_users = max(1, ceil((n_rows_target / max(1.0, avg_cnt)) * RANDOM_OVERSAMPLE)) if n_rows_target > 0 else 500

    sql = text(f"""
        WITH sample_users AS (
          SELECT "userId"
          FROM {MV_NAME}
          ORDER BY RANDOM()     -- new sample each run
          LIMIT :k_users
        )
        SELECT r."userId", r."movieId", r.rating, r."timestamp"
        FROM ratings r
        JOIN sample_users su USING ("userId");
    """)

    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn, params={"k_users": k_users})

    # If we undershoot badly (rare), bump K once
    if n_rows_target > 0 and len(df) < int(TARGET_FLOOR_RATIO * n_rows_target):
        k_users = int(ceil(k_users * UNDERSHOOT_RETRY_FACTOR))
        with engine.connect() as conn:
            df = pd.read_sql_query(sql, conn, params={"k_users": k_users})

    return df
