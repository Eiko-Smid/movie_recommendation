import streamlit as st
import os

# Base directory to locate images relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Movie Recommender Demo", layout="wide")
st.title("🎬 Movie Recommendation Project")

st.sidebar.success("Select a page above to navigate through the demo.")


col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("""
    Welcome to our interactive presentation! 👋 

    Use the sidebar to explore:
    - The dataset overview
    - The data pipeline
    - Details about our ALS model
    - How our API endpoints work
    """)
with col2:
    movie_img = os.path.join(base_dir, "utils/alex-litvin-MAYsdoYpGuk-unsplash.jpg")
    st.image(movie_img, use_container_width=False)


col3, col4 = st.columns([2, 3])
with col3:
    st.write("*Eiko Smid, Viktor Zivkov & Faith Osu Walter*")

with col4:
    movie_img2 = os.path.join(base_dir, "utils/denise-jans-Lq6rcifGjOU-unsplash.jpg")
    st.image(movie_img2, use_container_width=False)