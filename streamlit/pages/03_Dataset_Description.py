import os
import requests
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

import streamlit as st


# Apply global text alignment and formatting
st.markdown(
    """
    <style>
    /* Justify all paragraph and markdown text */
    p, li {
        text-align: justify !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("The MovieLens 20M Dataset🎞️")
st.markdown("""
The **MovieLens 20M** dataset is a stable benchmark dataset for building and evaluating movie recommendation systems.
It includes about 20 million user ratings applied to ~27,000 movies by ~138,000 users between 1995 and 2015. In this
project, we focus on  two  key tables:
            
🎬**movies.csv** contains movie metadata, including:
            
+ *movieId*: unique identifier for each movie
+ *title*: movie title and year of release
+ *genres*: pipe-seperated list of assigned genres (e.g., Action|Adventure|Sci-Fi)

📊**ratings.csv** holds all user-movie interactions with timestamped ratings (0.5-5.0 star range):
            
+ *userId, movieId, rating, timestamp (UNIX format)*
            
Other tables in the dataset (**tags.csv, links.csv, genome-tags.csv, genome-scores.csv**) provide tag, link, and feature metadata.
Although not used in the scope of this project, these can be integrated in a different setting for advanced analyses or content-based filtering.           
""")


@st.cache_data(persist=True) 
def get_rating_distribution() -> list[dict] | None:
    try:
        response = requests.get(
            url="http://api:8000/analytics/rating-distribution",
            timeout=10
        )
        
        # Raise error if status code is not 200
        response.raise_for_status()
        return response.json()
    
    except requests.RequestException as e:
        st.error(f"\nFailed to load rating distribution:\n{e}")
        return None


@st.cache_data(persist=True)
def get_db_information() -> dict | None:
    try:
        response = requests.get(url="http://api:8000/analytics/db-information")
        
        # Raise error if status code is not 200
        response.raise_for_status()
        return response.json()
    
    except requests.RequestException as e:
        st.error(f"\nFailed to load database information:\n{e}")
        return None


@st.cache_data(persist=True)
def get_ratings_head(n: int = 10) -> int | None:
    try:
        response = requests.get(
            url="http://api:8000/analytics/ratings-head",
            params={"n": n},
            timeout=5,
        )
        
        # Raise error if status code is not 200
        response.raise_for_status()
        return response.json()
    
    except requests.RequestException as e:
        st.error(f"\nFailed to load ratings table head:\n{e}")
        return None


@st.cache_data(persist=True)
def get_movies_head(n: int = 10) -> int | None:
    try:
        response = requests.get(
            url="http://api:8000/analytics/movies-head",
            params={"n": n},
            timeout=5,
        )
        
        # Raise error if status code is not 200
        response.raise_for_status()
        return response.json()
    
    except requests.RequestException as e:
        st.error(f"\nFailed to load movies table head:\n{e}")
        return None


# Get heads
movies_head = get_movies_head(n=10) or 1
ratings_head = get_ratings_head(n=10) or 1

# Get and extract DB informations
infos = get_db_information()
if infos:
    num_movies = infos.get("num_movies", 0)
    num_ratings = infos.get("num_ratings", 0)
    num_users = infos.get("num_users", 0)
    avg_rating = infos.get("avg_rating", 0)
    avg_ratings_per_user = infos.get("avg_ratings_per_user", 0)
    avg_ratings_per_movie = infos.get("avg_ratings_per_movie", 0)
    sparsity = infos.get("sparsity", 0)
else:
    num_movies = num_ratings = num_users = 0
    avg_rating = avg_ratings_per_user = avg_ratings_per_movie = sparsity = 0

# Gte rating distribution
rating_dist = get_rating_distribution()


# Define columns
col1, col2 = st.columns(2)

# Display movies head
with col1:
    st.subheader("Movies Table")
    st.dataframe(movies_head)
    st.caption("Contains movie metadata like titles and genres.")


# Display ratings head
with col2:
    st.subheader("Ratings Table")
    st.dataframe(ratings_head)
    st.caption("Shows users' movie ratings and timestamps.")


# Define columns
st.markdown("--------")
col3,col4,col5= st.columns(3)

# Display quick stats
with col3:
    st.subheader("QUICK STATS")
    st.metric("Number of Movies", f"{num_movies}") 
    st.metric("Number of Ratings", f"{num_ratings}")
    st.metric("Number of Users", f"{num_users}")
    st.caption("Summary of key dimensions, showing how many movies, ratings and unique users are included.")


# Display average vals and spars
with col4:
    # Additional metrics
    st.subheader("Average values and Sparsity")
    
    st.metric("⭐ Average rating", f"{avg_rating:.2f}")
    st.metric("👥 Average ratings per user", f"{avg_ratings_per_user:.2f}")
    st.metric("🎬 Average ratings per movie", f"{avg_ratings_per_movie:.2f}")
    st.metric("❌ Sparsity (missing ratings)", f"{sparsity:.2%}")
    st.caption("The high sparsity indicates that users rate only a small fraction of all movies. This is quite typical for such recommendation datasets.")


# Display rating distribution
with col5:
    st.subheader("Ratings Distribution")
    if rating_dist:
        # normal plotting
        ratings = [datapoint["rating"] for datapoint in rating_dist]
        counts = [datapoint["count"] for datapoint in rating_dist]

        fig, ax = plt.subplots()

         # Plot bar chart
        ax.bar(ratings, counts, width=0.4)

        # Axis labels
        ax.set_xlabel("Rating Score (0.5 - 5.0)", fontsize=10)
        ax.set_ylabel("Number of Ratings", fontsize=10)

        # Title
        ax.set_title("Distribution of User Ratings", fontsize=12, fontweight="bold")

        # Improve x-axis ticks (show all rating values clearly)
        ax.set_xticks(ratings)

        # Grid for readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Render plot
        st.pyplot(fig)
        
        # add description
        st.caption("Histogram showing how user ratings are distributed across the 0.5-5.0 star scale. This highlights typical rating preferences. The rating scores mostly cluster between 3 and 5 stars, reflecting positive user preferences.")
        
    else:
        
        st.warning("No data available for rating distribution")