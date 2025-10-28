import streamlit as st
import os


# Base directory to locate images relative to this script
base_dir = os.path.dirname(os.path.abspath(__file__))

st.title("MLOps Endpoints Overview")

# Endpoint: health
st.header("Endpoint: health")
st.write("Called: when user wishes to get infos")
st.markdown("""
This endpoint provides health information about the backend services, such as PostgreSQL and MLflow, 
to confirm that all critical components are operational.
""")

health_svg = os.path.join(base_dir, "../utils/API_health.svg")
st.image(health_svg, caption="Health Endpoint Overview", use_container_width=False)

# Endpoint: refresh-mv
st.header("Endpoint: refresh-mv")
st.write("Called: Once a day at 1:55 UTC")
st.markdown("""
This endpoint refreshes the materialized view in the database to filter for users with 5 or more ratings.  
The resulting table contains all user IDs meeting this criterion, ensuring the training data is relevant and up-to-date.
""")

refresh_mv_svg = os.path.join(base_dir, "../utils/API_refresh.svg")
st.image(refresh_mv_svg, caption="Refresh MV Endpoint", use_container_width=False)

# Endpoint: train
st.header("Endpoint: train")
st.write("Called: Once a day at 2:00 UTC")
st.markdown("""
The `/train` endpoint loads data from the refreshed materialized view, prepares CSR matrices, and performs a three-stage grid search to maximize MAP@K.  
It trains a new ALS model, logs artifacts to MLflow, compares with the current Champion, and promotes a better model to production.
""")

train_svg = os.path.join(base_dir, "../utils/API_train.svg")
st.image(train_svg, caption="Train Endpoint", use_container_width=False)

# Endpoint: recommend
st.header("Endpoint: recommend")
st.write("Called: when user wishes to get recommendations")
st.markdown("""
This endpoint loads the most recent Champion model from MLflow and generates personalized movie recommendations in real time.
""")

recommend_svg = os.path.join(base_dir, "../utils/API_recommend.svg")
st.image(recommend_svg, caption="Recommend Endpoint", use_container_width=False)
