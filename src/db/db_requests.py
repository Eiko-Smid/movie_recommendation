import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv 

from math import ceil

from fastapi import HTTPException, status
import pandas as pd

from src.db.database_session import engine


# Database settings
MV_NAME = "user_filter"          # name of materialized view
MIN_N_USER_RATINGS = 6              # users must have >=MIN_N_USER_RATINGS ratings to be part of MV
ASSUMED_AVG_PER_USER = 50.0         # rough avg ratings per eligible user (for sizing K)
RANDOM_OVERSAMPLE = 1.10            # oversample users by ~10% to hit target size
TARGET_FLOOR_RATIO = 0.9            # accept result if >=90% of n_rows
UNDERSHOOT_RETRY_FACTOR = 1.5       # if we undershoot badly, bump K by 50% once
AUTO_REFRESH_MV = False             # keep False to stay within +20% runtime budget


def _mv_exists() -> bool:
    '''
    Checks if the materialized view of the filtered users already exists.
    '''
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


def _ensure_indexes() -> None:
    '''
    Creates some useful indexes.
    '''
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{MV_NAME}_userId
            ON {MV_NAME}("userId");
        """))
        # Optional but useful
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_ratings_userid
            ON ratings("userId");
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_app_ratings_userid
            ON app_ratings("userId");
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_app_ratings_movieid
            ON app_ratings("movieId");
        """))


# def create_uv_all_ratings():
#     '''
#     Create unified rating view. 
#     '''
#     with engine.begin() as conn:
#         conn.execute(text(f'''
#             CREATE OR REPLACE VIEW all_ratings AS
#             SELECT
#                 ('ml_' || "userId"::text) AS user_key,
#                 "movieId",
#                 rating,
#                 "timestamp"
#             FROM ratings
#             UNION ALL
#             SELECT
#                 ('app_' || "userId"::text) AS user_key,
#                 "movieId",
#                 rating,
#                 "timestamp"
#             FROM app_ratings;
#         '''))


def create_uv_all_ratings() -> None:
    """
    Create unified ratings view with integer "userId".
    App users are shifted by (max MovieLens userId + 1).
    """
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE OR REPLACE VIEW all_ratings AS
            WITH max_index AS (
                SELECT COALESCE(MAX("userId"), 0) + 1 AS offset
                FROM ratings
            )
            SELECT
                r."userId"::bigint AS "userId",
                r."movieId",
                r.rating,
                r."timestamp"
            FROM ratings r
            UNION ALL
            SELECT
                (ar."userId"::bigint + m.offset) AS "userId",
                ar."movieId",
                ar.rating,
                ar."timestamp"
            FROM app_ratings ar
            CROSS JOIN max_index m;
        """))


def _create_mv_if_missing() -> None:
    if _mv_exists():
        return
    
    # Merge the ratings table with the registered user ratings (app_ratings) table.
    create_uv_all_ratings()    

    # Create MV (run once; safe to call every time)
    # Use a TX for CREATE; REFRESH CONCURRENTLY can't be in a TX, so we do not refresh here.
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE MATERIALIZED VIEW IF NOT EXISTS {MV_NAME} AS
            SELECT "userId"
            FROM all_ratings
            GROUP BY "userId"
            HAVING COUNT(*) >= :min_r;
        """), {"min_r": MIN_N_USER_RATINGS})
    _ensure_indexes()


def refresh_mv() -> bool:
    """
    Creates or refreshes the MV that stores all eligible user ids.
    (users with >= MIN_N_USER_RATINGS ratings across ratings + app_ratings). Also 
    creates the all_ratings view which is a combination of the table ratings and 
    app_ratings. "ratings" includes the ratings from MovieLense 20M dataset. The 
    app_ratings contains the ratings from the registered user. This data is evolving
    over time.
    """

    # Merge the ratings table with the registered user ratings (app_ratings) table.
    create_uv_all_ratings()

    # Create MV -> Only users with more than min_r exists 
    with engine.begin() as conn:
        # Ensure MV exists
        conn.execute(text(f"""
            CREATE MATERIALIZED VIEW IF NOT EXISTS {MV_NAME} AS
            SELECT "userId"
            FROM all_ratings
            GROUP BY "userId"
            HAVING COUNT(*) >= :min_r;
        """), {"min_r": MIN_N_USER_RATINGS})

        # Unique index needed for REFRESH CONCURRENTLY
        conn.execute(text(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{MV_NAME}_userId
            ON {MV_NAME}("userId");
        """))

    # 2) Refresh MV (concurrent if possible)
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text("SELECT pg_try_advisory_lock(987654321);"))
            conn.execute(text(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {MV_NAME};"))
            conn.execute(text("SELECT pg_advisory_unlock(987654321);"))
        return True
    except Exception:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(f"REFRESH MATERIALIZED VIEW {MV_NAME};"))
        return False


def _load_full_histories_for_n_users(n_users_target: int) -> pd.DataFrame:
    """
    Sample EXACTLY 'n_users_target' random eligible users (>=6 ratings) and return ALL of their ratings.
    """
    if n_users_target <= 0:
        raise ValueError("n_users_target must be > 0")

     # Ensure MV exists (and view)
    _create_mv_if_missing()

    # Check MV size (optional but helpful for a clear error)
    with engine.connect() as conn:
        mv_cnt = pd.read_sql_query(
            text(f'SELECT COUNT(*) AS c FROM {MV_NAME};'), conn
        ).iloc[0, 0]

    # If mv size is lower than desired n_users -> use mv size    
    if mv_cnt < n_users_target:
        n_users_target = mv_cnt

    # Extract all users n_users_target from the MV
    sql = text(f"""
            WITH sample_users AS (
                SELECT "userId"
                FROM {MV_NAME}
                ORDER BY RANDOM()     -- different set every run
                LIMIT :k_users        -- EXACT user count
            )
            SELECT r."userId", r."movieId", r.rating, r."timestamp"
            FROM all_ratings r
            JOIN sample_users su USING ("userId");
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn, params={"k_users": int(n_users_target)})

    return df



# def _sample_full_histories(n_rows_target: int) -> pd.DataFrame:
#     """
#     Sample K random eligible users and fetch ALL their ratings.
#     Targets ~n_rows_target rows overall (not exact, because full histories vary).
#     """
#     avg_cnt = _estimate_avg_per_user(engine)
#     k_users = max(1, ceil((n_rows_target / max(1.0, avg_cnt)) * RANDOM_OVERSAMPLE)) if n_rows_target > 0 else 500

#     sql = text(f"""
#         WITH sample_users AS (
#           SELECT "userId"
#           FROM {MV_NAME}
#           ORDER BY RANDOM()     -- new sample each run
#           LIMIT :k_users
#         )
#         SELECT r."userId", r."movieId", r.rating, r."timestamp"
#         FROM ratings r
#         JOIN sample_users su USING ("userId");
#     """)

#     with engine.connect() as conn:
#         df = pd.read_sql_query(sql, conn, params={"k_users": k_users})

#     # If we undershoot badly (rare), bump K once
#     if n_rows_target > 0 and len(df) < int(TARGET_FLOOR_RATIO * n_rows_target):
#         k_users = int(ceil(k_users * UNDERSHOOT_RETRY_FACTOR))
#         with engine.connect() as conn:
#             df = pd.read_sql_query(sql, conn, params={"k_users": k_users})

#     return df