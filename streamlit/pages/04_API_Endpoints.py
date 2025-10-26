import streamlit as st
import requests

st.title("Our API Endpoints")

st.header("🩺System Health Check")
st.markdown(" *Check if the database and MLflow server are reachable.* ")

if st.button("Run Health Check"):
    with st.spinner("Checking backend and MLflow server health..."):
        try:
            response = requests.get("http://api:8000/health")  # call FastAPI health endpoint
            if response.status_code == 200:
                report = response.json()
                st.success("All systems operational ✅")
                st.json(report)
            else:
                st.error("Some systems appear to be down ❌")
                st.json(response.json())
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to API: {e}")

    

st.header("🏋🏽Train the Model")
train_data = {
    "n_rows": 10000,
    "pos_threshold": 4.0,
    "als_parameter": {
        "bm25_K1_list": [100, 200],
        "bm25_B_list": [0.8, 1.0],
        "factors_list": [128, 256],
        "reg_list": [0.10, 0.20],
        "iters_list": [25]
    },
    "n_popular_movies": 100
}
if st.button("Start Training"):
    with st.spinner("Training in progress..."):
        response=requests.post("http://api:8000/train", json=train_data)
        if response.status_code == 200:
            st.success("Training completed!")
        else:
            st.error(f"Error:{response.text}")


st.header("⭐Get Recommendations")
col1, col2 = st.columns(2)

with col1:
    user_id = st.text_input("Enter User ID to have a movie recommended")

with col2:
    n_movies = st.number_input("Number of movies to recommend", min_value=1, max_value=100, value=5)

if st.button("Get Recommendations"):
    if user_id.isdigit():
        prediction_data = {
            "user_id": int(user_id),
            "n_movies_to_rec": n_movies,
            "new_user_interactions": [296, 318, 593]
        }    
        with st.spinner("Fetching recommendations...Hm, what could you like?🤔"):
            response=requests.post(f"http://api:8000/recommend", json=prediction_data)
            if response.status_code == 200:
                recommendations=response.json()
                st.write ("Recommendations:")
                titles = recommendations.get("movie_titles", [])
                genres = recommendations.get("movie_genres", [])

                for title, genre in zip (titles, genres):
                    st.write(f"-**{title}** _| Genres:_ {genre}")
            else:
                st.error(f"Error: {response.text}")
    else:
        st.error("Please enter a valid user ID (integer).")
