import streamlit as st
import pandas as pd

# Apply global text alignment and formatting
st.markdown(
    """
    <style>
    /* Justify all paragraph and markdown text */
    p, li {
        text-align: justify !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎬 Introduction to Movie Recommendation")
st.markdown("---")

# --- Function to display DataFrame with style ---
def show_dataframe(df, rows=None, cols=None):
    """Displays a styled dataframe with gray borders and light hover effect."""
    if rows is not None:
        df = df.head(rows)
    if cols is not None:
        df = df.iloc[:, :cols]

    styled = (
        df.style
        .set_table_styles([
            {'selector': 'table', 'props': [('border', '1px solid lightgray'), ('border-collapse', 'collapse')]},
            {'selector': 'th, td', 'props': [('border', '1px solid lightgray'), ('padding', '6px 8px')]},
            {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]},
            {'selector': 'tr:hover', 'props': [('background-color', '#eef6ff')]}
        ])
    )
    st.dataframe(styled, width="stretch")

# --- Introduction ---
st.header("📖 The history of movie recommendation")
st.markdown("""
Recommender systems are one of the oldest yet most relevant problems in machine learning. They have been used since
the 1990s to predict what items users might also like.  

An “item” can be anything:
- 🎬 Movies  
- 🎵 Music  
- 🛒 Products  
- 📰 News articles  

Early systems were simple — they suggested items based on similar users or similar content. A turning point came 
in **2006** with the **Netflix Prize**, a public competition to improve Netflix’s movie recommendation algorithm 
by 10% in RMSE, with a prize of **$1 million**.

Since then, recommender systems have become increasingly important and remain a major research field with huge economic
relevance:

- **Amazon** has reported that **35%** of its revenue comes from recommender systems.  
- **Netflix** says that **80%** of streamed content is chosen through recommendations.  

**Resources:**  
- [Amazon: Customer Experience and Recommendation Algorithms](https://www.customercontactweekdigital.com/customer-insights-analytics/news/amazon-algorithm-customer-experience)  
- [Wired: How Netflix’s Algorithms Work](https://www.wired.com/story/how-do-netflixs-algorithms-work-machine-learning-helps-to-predict-what-viewers-will-like/)
""")

# --- Task & Dataset ---
st.header("🎯 Task and Dataset")
st.markdown("""
The task is to create a **movie recommendation system** based on the **MovieLens 20M Dataset**, which contains **20 
million user ratings** for approximately **27,000 movies**.

Conceptually, this can be seen as a large **user–movie matrix**, where each **row** represents a user *u* and each **column** a movie *v*.

Because not every user rated every movie, many entries are **unknown** this makes the matrix **sparse**, which is one of the main challenges
in recommender systems.
""")

# --- Example Table using show_dataframe() ---
st.caption("📊 Example: User–Movie Rating Matrix")

data = {
    "Movie 1": [3, 4, 4, "..."],
    "Movie 2": ["x", 2, 1, "..."],
    "Movie 3": [4, "x", "x", "..."],
    "…": ["…", "…", "…", "…"]
}
df_example = pd.DataFrame(data, index=["User 1", "User 2", "User 3", "…"])
# Convert all entries to strings to avoid PyArrow conversion errors
df_example = df_example.astype(str)
show_dataframe(df_example)

st.caption("x = unknown rating")

# --- Goal ---
st.header("🎯 Goal")
st.markdown("""
1. **Fill the missing ratings** in the matrix (predict unknown values).  
2. **Recommend** for each user the *K* movies with the highest predicted ratings  
   that the user hasn’t rated yet.
""")

# --- Classical Approaches ---
st.header("📚 Classical Approaches")

st.subheader("👥 Collaborative Filtering")
st.markdown("""
- Assumes that **users with similar preferences** will like similar items.  
- Example: If user A likes movies 1, 2, 3, and 4 and user B likes 1, 2, 3 →  
  then it’s likely that B will also like movie 4.  
- Finds **patterns in the data itself** (user–item interactions).  
""")

st.subheader("🎞️ Content-Based Filtering")
st.markdown("""
- Assumes that **user preferences depend on item features** (e.g. genres, cast, release year).  
- Example: If a user likes comedy and action movies, they’ll probably like others with the same genres.  
- Doesn’t rely on similarities between users — only on the **content attributes**.
""")

# --- Modern Approaches ---
st.header("🚀 Modern Approaches")
st.markdown("""
Modern recommender systems may also use:
- 🧠 **Deep learning** (autoencoders, transformers, graph neural networks)  
- ⚡ **Real-time personalization** (contextual bandits, reinforcement learning)  
- 🧩 **Hybrid systems** combining content, collaborative, and contextual signals  

These generally perform better, but are **more complex** to implement.
""")

# --- Project Choice ---
st.header("✅ Project Choice")
st.markdown("""
For this project, the focus was on building a **robust MLOps pipeline**,  
not achieving the absolute best metrics.  

Therefore, the **classical collaborative filtering** approach was chosen.
""")

# --- Requirements ---
st.header("🧩 Requirements")
st.markdown("""
- Medium model performance  
- Library with:
  - Built-in recommendation functionality  
  - Ability to compute meaningful metrics  
  - Scalability  
""")

# --- Why Collaborative Filtering ---
st.header("💡 Why Collaborative Filtering?")
st.markdown("""
- Easy to implement  
- Requires only the **user–item matrix**  
- The **Implicit** library provides the required infrastructure  
- Uses the **Alternating Least Squares (ALS)** algorithm  
""")

st.markdown("---")
st.success("📘 This page outlines the theoretical background and rationale behind the Movie Recommender project.")