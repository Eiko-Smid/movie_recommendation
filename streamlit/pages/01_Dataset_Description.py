import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from sqlalchemy import create_engine
from dotenv import load_dotenv


st.title("The MovieLens 20M Dataset🎞️")
st.markdown("""
The **MovieLens 20M** dataset is a stable benchmark dataset for building and 
evaluating movie recommendation systems.
It includes about 20 million user ratings applied to ~27,000 movies by ~138,000 users between 1995 and 2015.
In this project, we focus on  two  key tables:
            
🎬**movies.csv** contains movie metadata, including:
            
+ *movieId*: unique identifier for each movie
+ *title*: movie title and year of release
+ *genres*: pipe-seperated list of assigned genres (e.g., Action|Adventure|Sci-Fi)

📊**ratings.csv** holds all user-movie interactions with timestamped ratings (0.5-5.0 star range):
            
+ *userId, movieId, rating, timestamp (UNIX format)*
            
Other tables in the dataset (**tags.csv, links.csv, genome-tags.csv, genome-scores.csv**) provide tag, link, and feature metadata.
Although not used in the scope of this project, these can be integrated in a different setting for advanced analyses or content-based filtering.           
""")


# To create engine and load data - Load environment variables and DB connection URL
load_dotenv()
DB_URL = os.getenv('DB_URL')
if DB_URL is None:
    raise ValueError("DB_URL environment variable is not set. Check your .env file and load_dotenv call.")

engine = create_engine(DB_URL)    

# Query the Database and read tables into pandas
query_movies = 'SELECT "movieId", title, genres FROM movies'
movies_df = pd.read_sql_query(query_movies, con=engine)  # or pd.read_sql(query_movies, con=engine)

query_ratings = 'SELECT "userId", "movieId", rating, timestamp FROM ratings'
ratings_df = pd.read_sql_query(query_ratings, con=engine)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Movies Table")
    st.dataframe(movies_df.head(10))
    st.caption("Contains movie metadata like titles and genres.")

with col2:
    st.subheader("Ratings Table")
    st.dataframe(ratings_df.head(10))
    st.caption("Shows users' movie ratings and timestamps.")


st.markdown("--------")
col3,col4,col5= st.columns(3)

with col3:
    st.subheader("QUICK STATS")
    st.metric("Number of Movies", f"{len(movies_df):,}") # 27,278
    st.metric("Number of Ratings",f"{len(ratings_df):,}") # 20,000,263
    st.metric("Number of Users","138,493")
    st.caption("Summary of key dimensions, showing how many movies, ratings and unique users are included.")

with col4:
    # Additional metrics
    st.subheader("            ")
    avg_rating = ratings_df['rating'].mean()
    avg_ratings_per_user = ratings_df.groupby('userId')['rating'].count().mean()
    avg_ratings_per_movie = ratings_df.groupby('movieId')['rating'].count().mean()
    total_possible = len(ratings_df['userId'].unique()) * len(ratings_df['movieId'].unique())
    sparsity = 1 - (len(ratings_df) / total_possible)

    st.metric("⭐ Average rating", f"{avg_rating:.2f}")
    st.metric("👥 Average ratings per user", f"{avg_ratings_per_user:.1f}")
    st.metric("🎬 Average ratings per movie", f"{avg_ratings_per_movie:.1f}")
    st.metric("❌ Sparsity (missing ratings)", f"{sparsity:.2%}")
    st.caption("The high sparsity indicates that users rate only a small fraction of all movies. This is quite typical for such recommendation datasets.")
  

with col5:
    st.subheader("Ratings Distribution")
    fig, ax = plt.subplots()
    ratings_sample = ratings_df.sample(n=30000)
    ratings_sample['rating'].hist(ax=ax, bins=10, color='cyan', edgecolor='black')
    ax.set_xlabel('Score')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    st.caption("Histogram showing how user ratings are distributed across the 0.5-5.0 star scale. This highlights typical rating preferences. The rating scores mostly cluster between 3 and 5 stars, reflecting positive user preferences.")