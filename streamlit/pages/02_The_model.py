import streamlit as st
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

# st.set_page_config(layout="wide")

# Add CSS to enable text wrapping and respect line breaks
st.markdown(
    """
    <style>
    .wrap-text td {
        white-space: pre-wrap !important;   /* enables \n line breaks */
        word-break: break-word !important;  /* wraps long words */
        text-align: left !important;        /* align text left */
        vertical-align: top !important;
    }
    .wrap-text th {
        text-align: center !important;
    }
    p, li {
        text-align: justify !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("🔢 Alternating Least Squares")
st.write(
    """
The goal of the algorithm is to fill in the missing ratings based on the given rating in the user-item matrix.
This matrix can easily be created from the ratings table, which we will see later in the "Data praperation" part.
Remember, in collaborative filtering we are assuming that users who like similar things, will have the same taste and therefore will have similar recommendations.
"""
)


data = {
    "Movie 1": [3, 4, 4, "..."],
    "Movie 2": ["x", 2, 1, "..."],
    "Movie 3": [4, "x", "x", "..."],
    "…": ["…", "…", "…", "…"]
}

st.caption("Example user–item ratings matrix (x = missing rating)")
df_example = pd.DataFrame(data, index=["User 1", "User 2", "User 3", "…"])
# Convert all entries to strings to avoid PyArrow conversion errors
df_example = df_example.astype(str)
show_dataframe(df_example)



st.write('''
The algorithm factorizes this huge matrix R into a matrix multiplication of the matrices called U and V^T:
''')

st.latex(r"\hat{R} = U \times V^{T}")

st.write(
    '''
By doing so the factorization finds the hidden linear patterns in the data itself. These hidden dimensions are called latent factors:

'''
)

st.markdown("""
By doing so the factorization finds the hidden linear patterns in the data itself. These hidden dimensions are called latent factors:

- If user 1 likes the movies 1 and 3, then ALS gives both similar coordinates in the hidden space
- If user 2 likes similar movies than user 1, their vectors in U become similar, too.
- In a n-dimensional space users with same preferences are close together.
            
The number of latent factors are typically defined manually. The higher the number, the more hidden patterns will be taken into account
but the higher the risk of overfitting. The shape and meaning of the matrices U and V goes hand in hand with the latent factors and 
hitten pattern in the data, as can be seen in the following table.

""")

V_and_U_table = {
    "Matrix": ["U", "V"],
    "Shape": ["m × f", "n × f"],
    "Meaning": [
        "User latent matrix. Each row represents one user as a vector of f latent factors.\n"
        "The user vector represents the taste and preferences of the user.",
        "Item latent matrix. Each row represents one item as a vector of f latent factors.\n"
        "The item vector represents the hidden characteristics of the item."
    ]
}

df = pd.DataFrame(V_and_U_table)
df = df.astype(str)
show_dataframe(df)


st.markdown(r"""
    Example: 
                
    Lets say we define f=3 latent factors.. Then user u might be represented as:

    $$
    U_u = [0.8, -0.1, 0.4]
    $$

    And item i is represented by the following vector:

    $$
    V_i = [0.9, -0.3, 0.5]
    $$
                
    If we wanne find out hwo much user u likes item i, than we have to multiply both vectors:
            
    $$
    \hat{r}_{ui} = U_u^{T} \cdot V_i = 0.8 \cdot 0.9 + (-0.1) \cdot (-0.3) + 0.4 \cdot 0.5 = 0.95
    $$

""")


st.header("Building the factorizaton")


st.markdown(r"""
The goal of the algorithm is to minimize a cost function such that the known ratings in the original dataset
gets approximated by the matrix multiplication as best as possible.  Indeed, the cost functions shows us
that we are trying to minimize the squared errors between the known ratings of the original matrix R and 
the predicted ratings $$(U_u^{T} \cdot V_i)$$. The second term is a regularization term which penalizes large values
in U and V, such that the ratings won't get too big.
""")

st.latex(r"""
J = \sum_{u,i} c_{ui} \left(p_{ui} - U_u^{T} V_i \right)^2
    + \lambda \left( \lVert U_u \rVert^2 + \lVert V_i \rVert^2 \right)
""")

st.markdown("""
K: Set of known user/item pairs\n
Lambda: regularization, Prevents overfitting by penalizing large latent vector norms\n
U: User-latent -matrix\n
V: item-latent -matrix\n
pui: 1 if user interacted with item i, else 0\n
cui: Confedence how mcuh we trust the interactions
            
The minimization of the cost function is done in two alternating steps:

Init matrices U and V with random values

	1) Fix matrix V and solve for U values with linear least squares
	2) Fix matrix U and solve for V with linear least squares


Because the two steps are alternating and the least square method is used iin each step, the method is 
verall called Alternating Leaast Squares.

""")

st.header("Recommend items")

st.markdown("""
Recommeding a mvoie for a user i simple, when the matrix factorization has been computed. Lets say that
our trained matrix (prediction ofmatrix R) is R^.  This matrix looks as follows now:
""")

data = {
    "Movie 1": [3, 4, 4, "..."],
    "Movie 2": [2, 2, 1, "..."],
    "Movie 3": [4, 3, 4, "..."],
    "…": ["…", "…", "…", "…"]
}

st.caption("Example user–item ratings matrix (x = missing rating)")
df_example = pd.DataFrame(data, index=["User 1", "User 2", "User 3", "…"])
# Convert all entries to strings to avoid PyArrow conversion errors
df_example = df_example.astype(str)
show_dataframe(df_example)

st.markdown("""
Now we can simply lookup and see that the user 3 has a value of 4 for movie 3 now, so we could recommend
this movie to him. For every user we can:

	1) Make a list of all ratings, he hadn't had before: 
            [(3, 4), (4, 5), (6, 4.5)]
	2) Sort this list from highest to lowest rating: 
            [(4, 5), (6, 4.5), (3, 4)] 
    3) Recommend the top 3 -> first 3 movies in the list -> movies 4, 6, 3
""")

