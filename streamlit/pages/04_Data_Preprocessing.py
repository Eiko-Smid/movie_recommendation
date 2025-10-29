import streamlit as st
import os
import pandas as pd

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


# Define csv path -> dir access
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_folder = "../preprocessing_steps/"
csv_dir = os.path.join(base_dir, csv_folder)

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


st.title("🧹 Dataset Preprocessing")
st.markdown("""
Here the data preprocessing steps are shown visually by displaying the data in tables after each preprocessing step.
Test data is used to demonstrate the preprocessing.
""")


st.markdown("""
### Step 0: Original Dataset
""")

# Load and display csv data
original_path = os.path.join(csv_dir, "original.csv")
df = pd.read_csv(original_path)
df = df.astype(str)
show_dataframe(df)
st.caption("Original data")


st.markdown("""
### Step 1: Drop NaN values
- Drop NaN values from the ratings data.
""")

# Load and display csv data
original_path = os.path.join(csv_dir, "nans.csv")
df = pd.read_csv(original_path)
df = df.astype(str)
show_dataframe(df)
st.caption("Cleared NaN values")


st.markdown("""
### Step 2: Convert Data Types
Ensure the columns have the correct datatypes for processing:
| Column    | Datatype |
|-----------|----------|
| userID    | int64    |
| movieId   | int64    |
| rating    | float32  |
| timestamp | int64    |
""")


st.markdown("""
### Step 3: Keep Only Ratings Greater than Threshold
- For implicit ALS, only positive ratings are used.
- We set a threshold of 4, keeping only ratings > 4.
""")

# Load and display csv data
original_path = os.path.join(csv_dir, "df_pos.csv")
df = pd.read_csv(original_path)
df = df.astype(str)
show_dataframe(df)
st.caption("Data > threshold")


st.markdown("""
### Step 4: Group by Latest Timestamp → Test Data
- Use the newest timestamps for the test set.
- People’s preferences change over time; training on older data and testing on newer data captures this.
""")


st.markdown("""
### Step 5: Train/Test Split
- Keep only users with at least one train and one test rating.
- Split data into training and testing sets.
""")

# Load and display csv data
original_path = os.path.join(csv_dir, "train_df.csv")
df = pd.read_csv(original_path)
df = df.astype(str)
show_dataframe(df)
st.caption("Training data")

original_path = os.path.join(csv_dir, "test_df.csv")
df = pd.read_csv(original_path)
df = df.astype(str)
show_dataframe(df)
st.caption("Test data")


st.markdown("""
### Step 6: Transform Data to CSR Matrices
- Convert data into sparse CSR matrices for efficient computation.
""")

# Load and display csv data
original_path = os.path.join(csv_dir, "train_csr.csv")
df = pd.read_csv(original_path)
df = df.astype(str)
show_dataframe(df)
st.caption("Train csr matrix")

# Load and display csv data
original_path = os.path.join(csv_dir, "test_csr.csv")
df = pd.read_csv(original_path)
df = df.astype(str)
show_dataframe(df)
st.caption("Test csr matrix")


st.markdown("""
### Step 8: Keep only users with more than 5 train entries and 1 test entrie 
- The metrices we are using, always check how relevant the recommendations are among top 10
- For a user with 1 train and 1 test entrie, the metric would be bade, because we would only have 1 hit among the 10 hits we want to have -> 1/10 = 0,1 
- Therfroe we are only using users with 5 train and 1 test entrie for teh evaluation part.
""")

# Load and display csv data
original_path = os.path.join(csv_dir, "test_filtered.csv")
df = pd.read_csv(original_path)
df = df.astype(str)
show_dataframe(df)
st.caption("Filtered test csr matrix")