import streamlit as st

st.set_page_config(page_title="Movie Recommender Demo", layout="wide")
st.title("🎬 Movie Recommendation Project")

st.sidebar.success("Select a page above to navigate through the demo.")

st.markdown(
    """
    Welcome to our interactive presentation! 👋 

    Use the sidebar to explore:
    - The dataset overview
    - The data pipeline
    - Details about our ALS model
    - How our API endpoints work
    """
)
