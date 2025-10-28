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

<img src="streamlit/utils/pipeline_refresh.svg" width="400" alt="Pipeline Flowchart"/>


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

data/dump
- dump.sql (backup of original DB)

.env (file)
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