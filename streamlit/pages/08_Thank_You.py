import streamlit as st
import os

# Base directory to locate images relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))

st.title("Thank You!")

movie_img = os.path.join(base_dir, "../utils/tyson-moultrie-BQTHOGNHo08-unsplash.jpg")
st.image(movie_img, use_container_width=False)