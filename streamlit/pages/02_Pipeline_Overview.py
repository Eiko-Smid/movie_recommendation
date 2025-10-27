import streamlit as st
import os

st.title("🎬 **Movie Recommendation System MLOps Pipeline**")

st.markdown('''
This pipeline implements a complete MLOps workflow for a movie recommendation system based on the ALS (Alternating Least Squares) algorithm. It is designed for reproducibility, automation, and continuous model improvement within an isolated Docker Compose environment.
 
The system connects several core components: **PostgreSQL**, **FastAPI**, **MLflow**, **Streamlit** and a **Cronjob** that automates retraining and model promotion.
 
**PostgreSQL** stores the processed MovieLens dataset (`movielens_db`), providing a consistent source for both training and inference data.  
 
**FastAPI** handles the backend logic with three main endpoints:  
  - `/train`: Executes model training. Triggerd by **Cronjob** *daily at 2:00UTC*, this endpoint fetches the latest data, trains a new ALS model (Challenger), logs artifacts and metrics in MLflow and compares it to the current Production Model (Champion).
              If the Challenger performs better, it t is promoted to production and becomes the new Champion in MLflow. 
    
--> By this means **Cronjob** enforces continuous training and model evaluation every day, enabling the system to adapt automatically and prevent model drift.

  - `/recommend`: Serves real-time recommendations by using the latest Champion model from MLflow and generating user-specific movie suggestions.  
  - `/health`: Checks the connectivity and status of PostgreSQL and MLflow to ensure all critical services are operational.  

**MLflow** manages experiment tracking, stores models, and governs the Champion/Challenger lifecycle, ensuring only the best model remains active in production.  

**Streamlit** functions both as an interactive **user interface** and as a **reference dashboard**, allowing users to request recommendations, review model outputs, and visualize key information about the system architecture and current state.   

This structure delivers a scalable, maintainable, and self-improving MLOps deployment for movie recommendations.
''')


base_dir = os.path.dirname(os.path.abspath(__file__))
svg_path = os.path.join(base_dir, "../utils/pipeline.svg")

st.image(svg_path)