-- Inits the Database. 
-- Gets called automatically. The containers build in logic checks if "PGDATA" folder is empty
-- if yes all files inside "/docker-entrypoint-initdb.d/" will be run automatically in alphabetic order.
-- Use same column names as CSV headers to simplify COPY. Put special ones like "movieId" in quotes 
-- otherwise postgres will automatically lower case them to: movieid and then it ma yenter problem in code.

-- # Create movies table to store movie info
CREATE TABLE IF NOT EXISTS movies (
    "movieId"   INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    genres      TEXT NOT NULL
);

-- # Create ratings table to store ratings info
CREATE TABLE IF NOT EXISTS ratings (
    "userId"    INTEGER NOT NULL,
    "movieId"   INTEGER NOT NULL REFERENCES movies("movieId") ON DELETE CASCADE,
    rating      NUMERIC(2,1) NOT NULL,
    "timestamp" BIGINT NOT NULL  -- quoted because it's a reserved word
);

-- Create type to ensure our column later on can only contain the predefined roles. 
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role') THEN
        CREATE TYPE user_role AS ENUM ('USER', 'DEVELOPER', 'ADMIN');
    END IF;
END$$;

-- Create user authentication and authorization table
CREATE TABLE IF NOT EXISTS users (
    id              BIGSERIAL PRIMARY KEY,          -- SQLAlchemy int PK; BIGSERIAL is common in Postgres
    email           VARCHAR(320) NOT NULL UNIQUE,    -- same 320 length, unique, not null
    hashed_password VARCHAR(255) NOT NULL,           -- same 255 length, not null
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,   -- default True, not null
    role            user_role NOT NULL DEFAULT 'USER'-- default USER
);

-- The ratings of the registered users from users table will we inside this table
-- This table will later be joined with ratings to get the overall training data
-- to train the model.
Create Table IF NOT EXISTS app_ratings (
    "userId"    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    "movieId"   INTEGER NOT NULL REFERENCES movies("movieId") ON DELETE CASCADE,
    rating      NUMERIC(2,1) NOT NULL,
    "timestamp" BIGINT NOT NULL,
    PRIMARY KEY ("userId", "movieId")
);

-- Create app_meta table which later be used to check if copy process of data is finished -> start api
CREATE TABLE IF NOT EXISTS app_meta (
    key TEXT Primary Key,
    value TEXT NOT NULL
);

-- Table to store general API configurations
CREATE TABLE IF NOT EXISTS api_config (
    key TEXT PRIMARY KEY,
    value BIGINT NOT NULL
)

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings("userId");
CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings("movieId");
CREATE INDEX IF NOT EXISTS idx_app_ratings_movie ON app_ratings("movieId");