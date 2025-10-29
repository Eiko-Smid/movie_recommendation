# Movie Recommendation Project
==============================

## Introduction
This project implements a movie recommendation system based on the widely used MovieLens 20 Million dataset. Recommender systems have been a key focus of machine learning since the 1990s, powering personalized suggestions across many domains like movies, music, products and news.

The main challenge lies in the *sparsity* of the user-item rating matrix. Out of over 3.7 billion possible user-movie pairs (138,000 users × 27,000 movies), only around 20 million ratings exist, meaning roughly 99.5% of the matrix is unknown.  
Sparsity explains why simple popularity-based or average rating methods often fail, pushing the field toward collaborative filtering and matrix factorization techniques. These techniques identify latent patterns and enable predictions even with very few observed interactions.  
It also motivates the use of *implicit feedback* methods, focusing on positive signals, to build more reliable recommendations.

## Dataset
- **movies.csv**: Contains movie metadata — unique movie IDs, titles, and pipe-separated genre lists (e.g., Action|Adventure|Sci-Fi).  
- **ratings.csv**: Holds timestamped user ratings in the 0.5 to 5.0 star range, recording userId, movieId, rating, and timestamp.

This annotation allows the system to understand both user behavior and item characteristics.

## Goal
1. **Predict missing ratings** in the user–movie matrix.  
2. **Recommend top K movies** with the highest predicted ratings to each user, focusing on movies the user has not yet rated.

## Collaborative Filtering Approach
We chose classical collaborative filtering via **Matrix Factorization** powered by the Alternating Least Squares (ALS) algorithm.
- The sparse user-item matrix \( R \) is factorized into two low-rank matrices:

$$
\hat{R} = U \times V^\top\
$$

where \( U \) encodes user latent features and \( V \) encodes item latent features.  
- These latent factors represent hidden preferences and characteristics, capturing patterns beyond explicit data.  
- ALS iteratively alternates between optimizing \( U \) and \( V \) by minimizing a cost function that balances fit to the data and regularization to prevent overfitting.  
- The number of latent factors controls model complexity: more factors capture richer patterns but risk overfitting.

## Why This Approach?
- **Scalable and robust**, handling large, sparse datasets efficiently.  
- Works well with *implicit feedback*, using confidence weights to reflect interaction certainty.  
- Well-supported by mature libraries (e.g., Implicit) that simplify implementation and evaluation.  
- Balanced trade-off between predictive performance and complexity, ideal for building a reliable MLOps pipeline.
---


Project Organization
------------
    sep25_bmlops_int_movie_reco/
    │
    ├── .github/workflows
    │   └── python-app.yml
    │
    ├── .venv/...
    │
    ├── .vscode/
    │   └── launch.json
    │
    ├── champ_store/
    │   └── champion_train_csr.npz
    │   
    ├── data/
    │   ├── dump/...
    │   ├── ml-20m/...
    │   └── preprocessing_steps/...
    │
    ├── logs/
    │      
    ├── mlflow/
    │   ├── mlartifacts\1/...
    │   ├── mlruns
    │   ├── .gitkeep
    │   └── mlflow.db
    │
    ├── models/...
    │
    ├── notebooks/...
    │
    ├── postgres_data/...
    │
    ├── references/...
    │        
    ├── reports/...
    │
    ├── src/ 
    │   ├── __pycache__/...
    │   ├── api/...      
    │   ├── data/...
    │   ├── features/...
    │   ├── models/...         
    │   ├── visualization/...
    │   ├── __init__.py
    │   └── config     
    │
    │ 
    ├── streamlit/ 
    │   ├── drawios/...       
    │   ├── pages/...      
    │   ├── utils/...    
    │   └── streamlit_app.py
    │
    ├── streamlit_cache/...
    │  
    ├── .env
    │  
    ├── .gitignore
    │  
    ├── docker-compose.yml
    │  
    ├── Dockerfile.api    
    │
    ├── Dockerfile.mlflow
    │  
    ├── Dockerfile.streamlit
    │  
    ├── LICENSE
    │
    ├── README.md
    │  
    ├── requirements.in   
    │  
    ├── requirements.txt 
    │  
    ├── terminal_out.txt
    │  
    └── train_payload.json



## Main Folders

- **data/**: Contains raw datasets, processed data, and intermediate files used during preprocessing and modeling.
- **src/**: Source code for data loading, feature engineering, model development, and utility functions.
- **streamlit/**: Streamlit app scripts and static assets like images, SVGs, and visualizations for interactive demos.
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

# Architecture and Pipeline

![Pipeline Flowchart](streamlit/utils/pipeline_refresh.svg)

The project is realized as a multi-container app. It consists of 5 Docker containers which are defined and configured with a docker-compose. The 5 containers are:
- **postgres_db:** Contains the database with the data needed to train the model. The data originates from the Movielens Database. It is used by the streamlit_app and the movie_rec_api container to provide their services.
- **mlflow_server:** Runs the MLflow server which tracks the experiments and runs, as well as model metrics and artifacts
- **movie_rec_api:** Hosts the API Endpoints
  - /health: Checks the readiness of the mlflow server and the database
  - /refresh-mv: Refreshes the materialized view. Database is updated with newest data
  - /train: Fetches data, trains the model, stores artifacts in mlflow, and sets current best model as production model
  - /recommend: Generates predictions based on the current production model
- **streamlit_app:** Contains a streamlit server for demo/presentation purposes as well as for providing a UI
- **daily_trainer:** Contains two cronjob tasks
  - Database refresh: Triggers the /refresh-mv endpoint daily at 1:55
  - Model training: Triggers the /train endpoint daily at 2:00


# Project setup

### 1. Clone repo

```bash
git clone https://github.com/DataScientest-Studio/sep25_bmlops_int_movie_reco.git
```
--------

### 2. Add files

Download and extract the Movielens DB from https://grouplens.org/datasets/movielens/20m/ and follow the next steps.

Folders to add:

data/ml-20m
- ratings.csv
- movies.csv

For the next step, run the script src/data/postgre_db_creation.py to generate a database from the csv's in ml-20m. Make a backup of the DB with the name dump.sql

data/dump
- dump.sql (backup of original DB)

.env file (adapt to your db-name, username, and password)
```
DB_URL=postgresql+psycopg2://postgres:Dbzices##01@postgres:5432/movielens_db
POSTGRES_DB=movielens_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
```

### 3. Build docker container

```bash
docker compose build 
```

or
```bash
docker compose build --no-cache
```
if build failed and you want clean new setup.

### 4. Start docker services

```bash
docker compose up
```

### 5. Start Debugger

- Go to vscode debugger (Run and Debug) in left sidebar
- Click on green play button

### 6. Wait until SQL DB is up

- If error in train endpoint happens, is mostly due to the fact that the DB isn't finished creating
- Just wait a few seconds and try again





-----------------------
## References and Further Reading
- [Amazon: Customer Experience and Recommendation Algorithms](https://www.customercontactweekdigital.com/customer-insights-analytics/news/amazon-algorithm-customer-experience)  
- [Wired: How Netflix’s Algorithms Work](https://www.wired.com/story/how-do-netflixs-algorithms-work-machine-learning-helps-to-predict-what-viewers-will-like/)  
- [Matrix Factorization and ALS Deep Dive](https://towardsdatascience.com/recsys-series-part-4-the-7-variants-of-matrix-factorization-for-collaborative-filtering-368754e4fab5/)
