import os

from typing import Optional

from sqlalchemy import create_engine, text, select, func
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

import logging

from dotenv import load_dotenv 

from math import ceil

from fastapi import HTTPException, status
import pandas as pd

from src.db.models.ratings import Rating
from src.db.database_session import engine, SessionLocal


# Database settings
MV_NAME = "user_filter"             # name of materialized view
MIN_N_USER_RATINGS = 6              # users must have >=MIN_N_USER_RATINGS ratings to be part of MV
USER_ID_OFFSET = None               

# ASSUMED_AVG_PER_USER = 50.0         # rough avg ratings per eligible user (for sizing K)
# RANDOM_OVERSAMPLE = 1.10            # oversample users by ~10% to hit target size
# TARGET_FLOOR_RATIO = 0.9            # accept result if >=90% of n_rows
# UNDERSHOOT_RETRY_FACTOR = 1.5       # if we undershoot badly, bump K by 50% once
# AUTO_REFRESH_MV = False             # keep False to stay within +20% runtime budget


# Define logger for logging
logger = logging.getLogger(__name__)


def get_user_id_offset() -> Optional[int]:
    '''
    Creates an SQLAlchemy session and returns the max user id of the ratings table. 
    Closes the session at the end. The id can then be used as an offset id value for
    another table, which includes independend user ids. 
    '''
    # Creates DB session
    session = SessionLocal()

    try:
        # Get the max user id of 
        stmt = select(func.max(Rating.userId))
        user_id_off: Optional[int] = session.scalar(stmt)
        return user_id_off
    except SQLAlchemyError:
        logger.exception("Failed to refresh the user id offset from ratings table.")
        raise
    finally:
        session.close()


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


def create_uv_all_ratings() -> None:
    """
    Create unified ratings view with integer "userId".
    App users are shifted by (max MovieLens userId + 1).
    """
    # Use max uid and fallback to 1 if not existing
    uid_offset = (get_user_id_offset() or 0) + 1

    # Union ratings with app_ratings
    sql = text("""
        CREATE OR REPLACE VIEW all_ratings AS
            SELECT
                r."userId"::bigint AS "userId",
                r."movieId",
                r.rating,
                r."timestamp"
            FROM ratings r
            UNION ALL
            SELECT
                (ar."userId"::bigint + :offset) AS "userId",
                ar."movieId",
                ar.rating,
                ar."timestamp"
            FROM app_ratings ar;        
    """)
    try:
        with engine.begin() as conn:
            conn.execute(sql, {"offset": uid_offset})
    except SQLAlchemyError:
        logger.exception("Failed to Union ratings with app_ratings table.")


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


def _load_full_mv_users() -> pd.DataFrame:
    # Check if MV exists and create if not
    if not _mv_exists():
        try:
            refresh_mv()
        except SQLAlchemyError as e:
            # DB-layer error: best treated as dependency/service issue (503)
            logger.exception("Failed to refresh/create MV '%s' due to DB error.", MV_NAME)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Materialized view '{MV_NAME}' is not available (DB error). Try again later.",
            ) from e
        except Exception as e:
            # Truly unexpected error: 500
            logger.exception("Unexpected error while refreshing/creating MV '%s'.", MV_NAME)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error while preparing materialized view '{MV_NAME}'.",
            ) from e
    
    # Extract all users n_users_target from the MV
    sql = text(f"""
            WITH sample_users AS (
                SELECT "userId"
                FROM {MV_NAME}
            )
            SELECT r."userId", r."movieId", r.rating, r."timestamp"
            FROM all_ratings r
            JOIN sample_users su USING ("userId");
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn)

    return df


def _load_full_histories_for_n_users(n_users_target: int) -> pd.DataFrame:
    """
    Sample EXACTLY 'n_users_target' randomly of the mv and return ALL of their ratings.
    """
    if n_users_target <= 0:
        raise ValueError("n_users_target must be > 0")
    
    # Check if MV exists and create if not
    if not _mv_exists():
        try:
            refresh_mv()
        except SQLAlchemyError as e:
            # DB-layer error: best treated as dependency/service issue (503)
            logger.exception("Failed to refresh/create MV '%s' due to DB error.", MV_NAME)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Materialized view '{MV_NAME}' is not available (DB error). Try again later.",
            ) from e
        except Exception as e:
            # Truly unexpected error: 500
            logger.exception("Unexpected error while refreshing/creating MV '%s'.", MV_NAME)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error while preparing materialized view '{MV_NAME}'.",
            ) from e

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
