Project Name
==============================

This project is a starting Pack for MLOps projects based on the subject "movie_recommandation". It's not perfect so feel free to make some modifications on it.

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
    │   ├── dump/
    │   │   └── dump.sql
    │   └── ml-20m/
    │       ├── genome-scores.csv
    │       ├── genome-tags.csv
    │       ├── links.csv
    │       ├── movies.csv
    │       ├── ratings.csv
    │       ├── README.txt
    │       └── tags.csv 
    │
    ├── logs/
    │
    ├── models/
    │   └── .gitkeep
    │
    ├── notebooks/
    │   └── .gitkeep
    │
    ├── postgres_data/
    │   ├── base/...
    │   ├── ...
    │   ├── ...
    │   └── postmaster.opts
    │
    ├── references/ 
    │   └── .gitkeep
    │        
    ├── reports/
    │   ├── figures/
    │       └── .gitkeep
    │   └── .gitkeep
    │
    ├── src/ 
    │   ├── __pycache__/
    │   │   └── __init__.cpython-311.pyc 
    │   │  
    │   │
    │   ├── api/       
    │   │   ├── __pycache__/
    │   │   │   ├── config.cpython-311.pyc 
    │   │   │   ├── movie_rec_api_and_mlflow_SQL.cpython-311.pyc 
    │   │   │   └── movie_rec_api_and_mlflow.cpython-311.pyc  
    │   │   │
    │   │   └── movie_rec_api_and_mlflow_SQL.py
    │   │   
    │   ├── data/
    │   │   ├── __pycache__/
    │   │   ├── mlartifacts\1/
    │   │   │   └── ...
    │   │   │   └── models/...
    │   │   │
    │   │   ├── mlruns/
    │   │   ├── __init__.py
    │   │   ├── db_requests.py
    │   │   ├── mlflow.db
    │   │   ├── postrgre_db_creation.py
    │   │   └── test_db_connection.py
    │   │
    │   ├── features/
    │   │   ├── __init__.py
    │   │   └── build_features.py
    │   │
    │   ├── models/         
    │   │   ├── __pycache__/
    │   │   │   ├── __init__.cpython-311.pyc 
    │   │   │   └── als_movie_rec.cpython-311.pyc 
    │   │   │
    │   │   ├── __init__.py
    │   │   └── als_movie_rec.py
    │   │
    │   ├── visualization/
    │   │   ├── __init__.py
    │   │   └── visualize.py
    │   │
    │   ├── __init__.py
    │   │
    │   └── config     
    │
    │ 
    ├── streamlit/ 
    │   ├── pages/      
    │   │   ├── 01_A_Bit_Of_History.py
    │   │   ├── 02_ALS_Model.py
    │   │   ├── 03_ALS_Model2.py
    │   │   ├── 04_Dataset_Description.py
    │   │   ├── 05_Pipeline_Overview.py
    │   │   ├── 06_API_Endpoints.py 
    │   │   ├── 07_Project_Outlook.py
    │   │   └── 08_Thank_You.py 
    │   │
    │   ├── utils/
    │   │   ├── pipeline_refresh.svg
    │   │   └── pipeline.svg
    │   │  
    │   └── streamlit_app.py
    │
    ├── streamlit_cache/...
    │  
    ├── .env
    │  
    ├── .gitignore
    │
    ├── 1_Training_before_test_set_filtering.txt   
    │  
    ├── 2_Training_after_test_set_filtering.txt 
    │  
    ├── 3_Training.txt
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
    ├── pipeline.drawio.svg
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