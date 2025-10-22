import streamlit as st
import requests

st.title("Our API Endpoints")
st.write("This is just a draft.")


st.header("Train the Model")
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
        response=requests.post("http://localhost:8000/train", json=train_data)
        if response.status_code == 200:
            st.success("Training completed!")
        else:
            st.error(f"Error:{response.text}")


st.header("Get Recommendations")
user_id=st.text_input("Enter User ID to have a movie recommended")

if st.button("Get Recommendations"):
    if user_id.isdigit():
        prediction_data = {
            "user_id": int(user_id),
            "n_movies_to_rec": 5,
            "new_user_interactions": [296, 318, 593]
        }    
        with st.spinner("Fetching recommendations...Hm, what could you like?🤔"):
            response=requests.post(f"http://localhost:8000/recommend", json=prediction_data)
            if response.status_code == 200:
                recommendations=response.json()
                st.write ("Recommendations:")
                for movie in recommendations.get("movie_titles", []):
                    st.write(f"-{movie}")
            else:
                st.error(f"Error: {response.text}")
    else:
        st.error("Please enter a valid user ID (integer).")
