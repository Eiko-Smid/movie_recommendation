# Movie Recommendation Project

## Description

This project is a **production-ready movie recommendation system** that provides movie recommendations to users based 
on other user preferences (**collaborative filtering**). After registration the user is able to rate movies and receive
recommendations. Even without any prior ratings, the system is able to generate meaningful initial recommendations through
a **cold-start strategy**.


## Features

The system is driven by a robust **MLOps pipeline** including the following features:
- **Streamlit frontend application** for system interaction and project presentation
- Real-time movie recommendations served via a **FastAPI-based API**
- Secured API using **JWT Authentication** and **Role-Based Access Control (RBAC)**
- Automated **model training and evaluation pipeline** with **MLflow**, ensuring that the best-performing model is deployed to production.
- **Monitoring** with **Prometheus** and **Grafana** to ensure model quality and system reliability 
- **CI pipeline** powered by **GitHUb Workflow**, including code linting/testing, container building/testing and pushing images to **DockerHub**
- **CD pipeline** that deploys the system on a predefined server, whenever a new software Release is created.


## System Architecture
### C4 Level 2: Container Architecture
![C4 Level 2: Container Architecture](streamlit/utils/Level_2_container_flowchart.svg)

### Short explanation
The system follows a container-based architecture powered by **Docker**. Interactions with the system are served by the
**Streamlit frontend**. This includes registration/login, rating movies and user-specific movie recommendations. All 
interactions are routed through the **FastAPI** backend which acts as the central component handling: 

- authentication/authorization
- user management
- model interaction (training and recommendation serving)
- database interactions  
- Monitoring metrics 

This architecture ensures strict **separation of concerns (SoC)** by enforcing all data and model access through the API layer.
All the data (user credentials, user ratings and the dataset) is stored inside the **PostgreSQL DB**. This container acts as the
central storage location of the system.

**Model training** is scheduled at predefined intervals by the **Training Scheduler**. To ensure a reproducible training, we are storing the training artifacts as well as the model inside **MLflow**.<br>
Finally the implemented **monitoring** ensures model quality and system reliability. Therefore the API provides metrics to the **Prometheus**
metrics server, while **Grafana** is reading these metrics from server and visualizes them.


## Quick Start Guide
### 0. Prerequisites

In order for the setup to work you need:
- Git (to clone the repository)
- Docker (incl. Docker Compose)


### 1. Clone repo

```bash
git clone https://github.com/Eiko-Smid/movie_recommendation.git <folder_name>
cd <folder_name>
```
--------

### 2. Add files

Download and extract the Movielens DB from https://grouplens.org/datasets/movielens/20m/ and follow the next steps.

Folders to add:

data/ml-20m
- ratings.csv
- movies.csv

Copy the following content inside a folder called .env, in the project root. Adapt to your preferences.
```
# PosgreSQL DB configuration
POSTGRES_DB=movielens_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
DB_URL=postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}

# JWT token params
JWT_SECRET=change-me-super-secret
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=45

# Admin Credentials
ADMIN_EMAIL=admin@company.com
ADMIN_PASSWORD=admin_password

# Define service token to give trainer access to train endpoint
API_SERVICE_TOKEN=change-me-service-token

# Define docker vars
DOCKER_USERNAME=eikosmid
```

### 3. Run containerized system

```bash
docker compose pull
docker compose up -d
```

### 4. Start Streamlit Application

Open http://localhost:8501/ in browser. Refrsh site once with "Strg + F5" -> Correct format shown

For API interaction go to page: "API Interaction"

Here u can register, login and then access the endpoint for "ACCESS_TOKEN_EXPIRE_MINUTES". After that logout and 
log in again. 

Note: Initial database setup may take several minutes.


# Project Structure
------------
    sep25_bmlops_int_movie_reco/
    │
    ├── .github/workflows               <- Github workflow defining the CI/CD pipeline (GitHub Actions)
    │
    ├── .vscode/                        <- Infos for python debugger to enable api debugging
    │
    ├── alembic                         <- Storage for champion model   
    │
    ├── champ_store/                    <- Storage for champion train csr matrix
    │   
    ├── data/                           <- Storage for dataset
    │
    ├── grafana/                        <- Grafana dashboard configs and datasources
    │      
    ├── logs/                           <- Training Scheduler logs
    │
    ├── mlflow/                         <- MLflow tracking & artifacts
    │
    ├── prometheus/                     <- Prometheus config
    │
    ├── src/                            <- Core backend logic (code)
    │   ├── api/                        <- FastAPI set up
    │   ├── db/                         <- Database 
    │   ├── models/                     <- Model training and interaction
    │   └── observability/              <- Monitoring 
    │ 
    ├── streamlit/                      <- Frontend application      
    │   ├── pages/                      <- UI pages
    │   ├── utils/                      <- Helpers
    │   └── streamlit_app.py            <- Streamlit starting page
    │
    ├── streamlit_cache/                <- Cached streamlit data to fasten up page loading
    │
    ├── tests                           <- pytest for CI pipeline
    │
    ├── trainer_utils                   <- Scripts and payload for Training Scheduler
    │    
    ├── .env                            <- Environment variables
    │  
    ├── docker-compose.yml              <- Container orchestration and Dockerfiles
    │  
    ├── Dockerfile.api                  
    │
    ├── Dockerfile.mlflow               
    │  
    ├── Dockerfile.streamlit            
    │  
    ├── LICENSE
    │
    ├── pyproject.toml                  <- Project definitions
    │
    ├── README.md
    |
    ├── requirements-dev.txt            <- Requirement files 
    │  
    ├── requirements.in
    │  
    └── requirements.txt            


## References and Further Reading
- [Amazon: Customer Experience and Recommendation Algorithms](https://www.customercontactweekdigital.com/customer-insights-analytics/news/amazon-algorithm-customer-experience)  
- [Wired: How Netflix’s Algorithms Work](https://www.wired.com/story/how-do-netflixs-algorithms-work-machine-learning-helps-to-predict-what-viewers-will-like/)  
- [Matrix Factorization and ALS Deep Dive](https://towardsdatascience.com/recsys-series-part-4-the-7-variants-of-matrix-factorization-for-collaborative-filtering-368754e4fab5/)
