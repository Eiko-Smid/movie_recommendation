import streamlit as st

st.title("Dataset Preprocessing")
st.write("The steps for creating the data needed for training the model are straightforward:")

st.markdown("""
### Step 0: Original Dataset
- Get original csv files
---
### Step 1: Drop NaN values
- Drop NaN values from the ratings data.

---
### Step 2: Convert Data Types
Ensure the columns have the correct datatypes for processing:
| Column    | Datatype |
|-----------|----------|
| userID    | int64    |
| movieId   | int64    |
| rating    | float32  |
| timestamp | int64    |
---
### Step 3: Keep Only Ratings Greater than Threshold
- For implicit ALS, only positive ratings are used.
- We set a threshold of 4, keeping only ratings > 4.

---
### Step 4: Group by Latest Timestamp → Test Data
- Use the newest timestamps for the test set.
- People’s preferences change over time; training on older data and testing on newer data captures this.

---
### Step 5: Train/Test Split
- Keep only users with at least one train and one test rating.
- Split data into training and testing sets.

---
### Step 6: Transform Data to CSR Matrices
- Convert data into sparse CSR matrices for efficient computation.

---
### Step 7: Keep Only Users with Sufficient Entries
- Keep users with more than 5 train entries and at least 1 test entry.
- Metrics evaluate recommendation relevance among the top 10.
- Users with too few entries would distort the metric results.
""")
