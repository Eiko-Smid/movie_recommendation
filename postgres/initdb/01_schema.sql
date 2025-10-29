-- Inits the Database. 
-- Gets called automnatically. The containers build in logic checks if "PGDATA" folder is empty
-- if yes all files inside "/docker-entrypoint-initdb.d/" will be run automatically in alphabetic order.
-- Use same column names as CSV headers to simplify COPY. Put special ones like "movieId" in quotes 
-- otherwise postgres will automatically lower case them to: movieid and then it ma yenter problem in code.
CREATE TABLE IF NOT EXISTS movies (
    "movieId"   INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    genres      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ratings (
    "userId"    INTEGER NOT NULL,
    "movieId"   INTEGER NOT NULL REFERENCES movies("movieId") ON DELETE CASCADE,
    rating      NUMERIC(2,1) NOT NULL,
    "timestamp" BIGINT NOT NULL  -- quoted because it's a reserved word
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings("userId");
CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings("movieId");
