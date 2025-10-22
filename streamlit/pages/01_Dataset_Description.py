import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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


# movies_preview = pd.read_csv ("data/ml-20m/movies.csv") [:10]
# ratings_preview = pd.read_csv ("data/ml-20m/ratings.csv") [:10]
movies = pd.read_csv ("data/ml-20m/movies.csv")
ratings = pd.read_csv ("data/ml-20m/ratings.csv")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Movies Table")
    st.dataframe(movies.head(10)) # or just use movies_preview
    st.caption("Contains movie metadata like titles and genres.")

with col2:
    st.subheader("Ratings Table")
    st.dataframe(ratings.head(10)) # or just use ratings_preview
    st.caption("Shows users' movie ratings and timestamps.")


st.markdown("--------")
col3,col4,col5= st.columns(3)

with col3:
    st.subheader("QUICK STATS")
    st.metric("Number of Movies", f"{len(movies):,}") # 27,278
    st.metric("Number of Ratings",f"{len(ratings):,}") # 20,000,263
    st.metric("Number of Users","138,493")
    st.caption("Summary of key dimensions, showing how many movies, ratings and unique users are included.")

with col4:
    # Additional metrics
    st.subheader("            ")
    avg_rating = ratings['rating'].mean()
    avg_ratings_per_user = ratings.groupby('userId')['rating'].count().mean()
    avg_ratings_per_movie = ratings.groupby('movieId')['rating'].count().mean()
    total_possible = len(ratings['userId'].unique()) * len(ratings['movieId'].unique())
    sparsity = 1 - (len(ratings) / total_possible)

    st.metric("⭐ Average rating", f"{avg_rating:.2f}")
    st.metric("👥 Average ratings per user", f"{avg_ratings_per_user:.1f}")
    st.metric("🎬 Average ratings per movie", f"{avg_ratings_per_movie:.1f}")
    st.metric("❌ Sparsity (missing ratings)", f"{sparsity:.2%}")
    st.caption("The high sparsity indicates that users rate only a small fraction of all movies. This is quite typical for such recommendation datasets.")
  

with col5:
    st.subheader("Ratings Distribution")
    fig, ax = plt.subplots()
    ratings_sample = pd.read_csv ("data/ml-20m/ratings.csv", nrows=50000)
    ratings_sample['rating'].hist(ax=ax, bins=10, color='cyan', edgecolor='black')
    ax.set_xlabel('Score')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    st.caption("Histogram showing how user ratings are distributed across the 0.5-5.0 star scale. This highlights typical rating preferences. The rating scores mostly cluster between 3 and 5 stars, reflecting positive user preferences.")