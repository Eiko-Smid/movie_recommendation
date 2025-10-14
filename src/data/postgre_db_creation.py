import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# --------------------------
# 1. Load environment variables
# --------------------------
load_dotenv()


# --------------------------
# 2. Database connection
# --------------------------
DB_URL = os.getenv("DB_URL")
engine = create_engine(DB_URL, echo=False)


# --------------------------
# 3. Helper: load and preprocess CSVs
# --------------------------
def load_csv(file_path, dtype=None):
    """Load CSV with UTF-8 encoding, fallback to latin-1 if needed."""
    try:
        df = pd.read_csv(file_path, dtype=dtype, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, dtype=dtype, encoding="latin-1")
    return df


def preprocess_dataframes(data_dir):
    """Load and preprocess MovieLens CSV files."""
    # Ratings
    ratings = load_csv(
        os.path.join(data_dir, "ratings.csv"),
        dtype={"userId": int, "movieId": int, "rating": float, "timestamp": int},
    )
    ratings.dropna(inplace=True)
    ratings["rating"] = ratings["rating"].astype(float)

    # Movies
    movies = load_csv(
        os.path.join(data_dir, "movies.csv"),
        dtype={"movieId": int, "title": str, "genres": str},
    )
    movies.dropna(inplace=True)

    # Tags
    tags = load_csv(
        os.path.join(data_dir, "tags.csv"),
        dtype={"userId": int, "movieId": int, "tag": str, "timestamp": int},
    )
    tags.dropna(subset=["tag"], inplace=True)

    # Links
    links = load_csv(
        os.path.join(data_dir, "links.csv"),
        dtype={"movieId": int, "imdbId": str, "tmdbId": str},
    )
    links.dropna(subset=["movieId"], inplace=True)

    # Genome scores
    genome_scores = load_csv(
        os.path.join(data_dir, "genome-scores.csv"),
        dtype={"movieId": int, "tagId": int, "relevance": float},
    )
    genome_scores.dropna(inplace=True)

    # Genome tags
    genome_tags = load_csv(
        os.path.join(data_dir, "genome-tags.csv"),
        dtype={"tagId": int, "tag": str},
    )
    genome_tags.dropna(inplace=True)

    return {
        "ratings": ratings,
        "movies": movies,
        "tags": tags,
        "links": links,
        "genome_scores": genome_scores,
        "genome_tags": genome_tags,
    }


# --------------------------
# 4. Insert into SQL database
# --------------------------
def insert_to_db(dfs: dict):
    """Insert DataFrames into SQL tables."""
    for name, df in dfs.items():
        print(f"Inserting {name} ({len(df)} rows)...")
        df.to_sql(name, engine, if_exists="replace", index=False)
    print("✅ All tables inserted successfully.")


# --------------------------
# 5. Main
# --------------------------
if __name__ == "__main__":
    # Get absolute directory of the current script file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # CSV files are in the same folder as the script, so just pass script_dir 
    data_dir = "data/ml-20m"

    # Confirmation
    print(f"Current script directory: {script_dir}")
    print(f"Loading data files from: {data_dir}")

    
    print("📥 Loading and preprocessing data...")
    dfs = preprocess_dataframes(data_dir)
    print("💾 Inserting into PostgreSQL database movielens_db...")
    insert_to_db(dfs)
