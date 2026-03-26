# Movie Recommendation Project

## Description
________________________________________________________________________________________________________________
This project is a **production-ready movie recommendation system** that provides movie recommendations to users based 
on other user preferences (**collaborative filtering**). After registration the user is able to rate movies and receive recommendations. Even without any prior ratings, the system is able to generate meaningful initial
recommendations through a **cold-start strategy**.


## Features
________________________________________________________________________________________________________________

The system is driven by a robust **MLOps pipeline** including the following features:
- **Streamlit frontend application** for system interaction and project presentation
- Real-time movie recommendations served via a **FastAPI-based API**
- Secured API using **JWT Authentication** and **Role-Based Access Control (RBAC)**
- Automated **model training and evaluation pipeline** with **MLflow**, ensuring that the best-performing model is
deployed to production.
- **Monitoring** with **Prometheus** und **Grafana** to ensure model performance and system reliability 
- **CI pipeline** powered by **GitHUb Workflow**, including code linting/testing, container building/testing and
pushing images to **DockerHub**
- **CD** pipeline that deploys the system on an predefined server, whenever a new software Release is created.



## System Architecture
### C4 Level 2: Container Architecture
![C4 Level 2: Container Architecture](streamlit/utils/Level_2_container_flowchart.svg)

### Short explanation
________________________________________________________________________________________________________________
The system follows a container-based architecture powered by **Docker**. Interactions with the system are served by
the **Streamlit frontend**. This includes registration/login, rating movies and user specific movie recommendations.
All interactions are routed through the **FastAPI** backend which acts as the central component handling: 

- authentication/authorization
- user management
- model inference (training and recommendation serving)
- database interactions  
- Monitoring metrics 

This architecture ensures strict separation of concerns (SoC) by enforcing all data and model access through the API layer.
All the data (user credentials, user ratings and the dataset) is stored inside the PostgreSQL DB. This container acts as
the central storage location of the system.

Model training is scheduled at predefined intervals by the **Training Scheduler**. To ensure a reproducible training, we are storing the training artifacts as well as the model inside **MLflow**.<br>
Finally the implemented monitoring ensures model quality and system reliability. Therefore the API provides metrics to the
Prometheus metrics server, while Grafana is reading these metrics from server and visualizes them.





## Quick Start Guide


## Authentication usage


## CI/CD


# Project Structure


# Documentation

Start Streamlit apllication after setup ...
