import streamlit as st
import os

st.title("🎬 **Movie Recommendation System MLOps Pipeline**")

st.markdown('''
This pipeline implements a complete MLOps workflow for a movie recommendation system based on the ALS (Alternating Least Squares) algorithm. It is designed for reproducibility, automation, and continuous model improvement within an isolated Docker Compose environment.
 
The system connects several core components: **PostgreSQL**, **FastAPI**, **MLflow**, **Streamlit** and two **Cronjobs** that automate data refresh and model retraining.
''')

with st.expander("PostgreSQL"):
    st.write("""
    **PostgreSQL** stores the processed MovieLens dataset (`movielens_db`),
    providing a consistent source for both training and inference data.
    """)

with st.expander("FastAPI"):
    st.write("""
    **FastAPI** handles the backend logic with four main endpoints:
              
    - `/refresh-mv`: Refreshes the materialized view in the database to filter out users with less than 5 ratings, ensuring that training data focuses only on users meeting this minimum threshold.
    This view acts as the updated source of data for model training. A **Cronjob** triggers this endpoint daily at **1:55 UTC**, preparing fresh data before training begins.
            
    - `/train`: Executes model training. Triggered daily at **2:00 UTC** by a **Cronjob**, this endpoint:
      - Loads ratings and movie data from the refreshed materialized view.
      - Prepares sparse CSR matrices for efficient ALS computation.
      - Runs an advanced, three-stage grid search to maximize MAP@K performance.
      - Trains a new ALS model (*Challenger*).
      - Logs artifacts and metrics in MLflow.
      - Compares the *Challenger* to the current Production Model (*Champion*).
      - Promotes the new model to production in MLflow, if the *Challenger* outperforms the *Champion*. 

    - `/recommend`: Serves real-time recommendations by using the latest *Champion* model from MLflow and generating user-specific movie suggestions.  
            
    - `/health`: Checks the connectivity status of PostgreSQL and MLflow to ensure all critical backend services are operational. 
    """)

with st.expander("MLflow"):
    st.write('''
    **MLflow** manages experiment tracking, stores models and governs the *Champion/Challenger* lifecycle, ensuring only the best model remains active in production.
    ''')

with st.expander("Streamlit"):
    st.write("""
    **Streamlit** functions both as an interactive **user interface** and a **reference dashboard**,
    allowing users to request recommendations, review model outputs and visualize key information about system architecture and current state. 
    """)
 
st.markdown('''   
Coordinating these components with two daily Cronjobs —one to refresh filtered training data at 1:55 UTC, and one to retrain and deploy models at 2:00 UTC— enforces continuous training and model evaluation,
enabling the system to adapt automatically and prevent model drift.
This architecture therefore, delivers a scalable, maintainable and self-improving MLOps deployment for movie recommendations.
''')


base_dir = os.path.dirname(os.path.abspath(__file__))
svg_path = os.path.join(base_dir, "../utils/pipeline_refresh.svg")

st.image(svg_path)