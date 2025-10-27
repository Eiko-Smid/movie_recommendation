import streamlit as st
import os

st.title("Our Pipeline")

base_dir = os.path.dirname(os.path.abspath(__file__))
svg_path = os.path.join(base_dir, "../utils/pipeline.svg")

st.image(svg_path)