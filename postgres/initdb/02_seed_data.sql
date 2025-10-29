-- postgres/initdb/03_seed_data.sql
-- WARNING: runs only on first DB init (empty PGDATA).

-- The CSVs will be mounted at /docker-entrypoint-initdb.d/seed
\echo 'Seeding movies...'
COPY public.movies("movieId", title, genres)
FROM '/seed/movies.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"');

\echo 'Seeding ratings...'
COPY public.ratings("userId", "movieId", rating, "timestamp")
FROM '/seed/ratings.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',', QUOTE '"');

SELECT COUNT(*) FROM "movies";
SELECT COUNT(*) FROM "ratings";
