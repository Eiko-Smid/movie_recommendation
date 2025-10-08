from __future__ import annotations
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple, Mapping, Iterable, Any, TypedDict


import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from implicit.evaluation import train_test_split, precision_at_k, mean_average_precision_at_k

from itertools import product




# _________________________________________________________________________________________________________
# Dataclasses
# _________________________________________________________________________________________________________

@dataclass
class Mappings():
    '''
    Stores the relevant mappings needed to transfer the df user/item ids to the user/item ids of 
    the csr matrices.
    '''
    user_index_to_id: Dict[int, int]
    item_index_to_id: Dict[int, int]
    user_id_to_index: Dict[int, int]
    item_id_to_index: Dict[int, int]


@dataclass
class ALS_Metrics():
    '''
    Metrics to compare the ALS model performance.
    '''
    prec_at_k: float
    map_at_k: float



# _________________________________________________________________________________________________________
# ALS functionality code
# _________________________________________________________________________________________________________

def get_popular_items(
        df: pd.DataFrame,
        top_n: int = 50,
        threshold: float = 4.0
        ) -> List[int]:
    '''
    This function will be frequently used to build a list containing the popular items.
    This popular items can the be used for the case that a new user needs to get movie
    recommendations. 
    '''
    # Keep only positives (same threshold as your training)
    df_pos = df[df["rating"] >= threshold]
    # Count per movieId
    counts = df_pos.groupby("movieId").size().sort_values(ascending=False)
    return counts.index.tolist()[:top_n]


# Build USER×ITEM matrices (binary)
def build_binary_coo(
        df_pos: pd.DataFrame,
        user_id_to_index: Mapping[int, int],
        item_id_to_index: Mapping[int, int],
        n_users: int,
        n_items: int
    ) -> coo_matrix:
    '''
    Transforms the given df into a binary coo matrix. It is therefore necessary that the
    given df only contains positive values. In the sense of ratings the df should only
    contain the part of the original data which is bigger than a threshold -> positives.
    All values are being transformed to one. 
    '''
    # Create mappings -> ids = [2, 4, 5, 9] -> [0, 1, 2, 3]
    uidx = df_pos["userId"].map(user_id_to_index).astype(np.int32).to_numpy()
    iidx = df_pos["movieId"].map(item_id_to_index).astype(np.int32).to_numpy()
    # Create data (all ones because all data is alreasy > threshold) -> only relevant data
    data = np.ones(len(df_pos), dtype=np.float32)
    # Build coo matrix
    return coo_matrix((data, (uidx, iidx)), shape=(n_users, n_items), dtype=np.float32)



def prepare_data(
        df: pd.DataFrame,
        pos_threshold: float = 3.0,
        ) -> Tuple[csr_matrix, csr_matrix, Mappings]:
    '''
    Prepare the data by dropping nons, keep only ratings > threshold, split into train and test and 
    build scr matrixes of them. Returns the train, test csr matrices and the mappings.
    '''
    print(f"\nOriginal df shape: {df.shape}")
    print(f"Original df:\n{df.head(20)}")

    # Clean data and convert types
    df = df.dropna(subset=["userId", "movieId", "rating"])
    df["userId"] = df["userId"].astype(np.int64)
    df["movieId"] = df["movieId"].astype(np.int64)
    df["rating"] = df["rating"].astype(np.float32)
    df["timestamp"] = df["timestamp"].astype(np.int64)

    # Display df infos
    print(f"\nShape after drop nans: {df.shape}")
    print(f"Data after drop nans:\n{df.head(10)}")

    # Keep only pos ratings
    df_pos = df.loc[df["rating"] >= pos_threshold].copy()
    if df_pos.empty:
        raise ValueError(f"No positives after binarization; lower pos_threshold < {pos_threshold} or data checking needed.")
    print(f"\nData after filtering by threshold {pos_threshold}: {df_pos.shape}")
    print(f"Data after filtering by threshold {pos_threshold}:\n{df_pos.head(10)}")
    
    # Here we wanne find the rows per user with the latest timestamp -> newest datas
    # This newest data will be used as test dataset. If we have multiple rows per user with same timestampe, all will be used as train data
    latest_ts = df_pos.groupby("userId")["timestamp"].transform("max")
    df_pos["is_test"] = (df_pos["timestamp"] == latest_ts)
    print(f"\nData shape after grouping by latest timestamp: {df_pos.shape}")
    print(f"Data after grouping by latest timestamp:\n{df_pos.head(10)}")

    # Users with at least one train and one test interaction
    # (some users may have only one positive; they will have no train or no test)
    # Keep tests as exactly the "latest" rows; if multiple ties on same timestamp, multiple test items possible.
    print("\nGroupe and transform data ...")
    grp = df_pos.groupby("userId")["is_test"]
    has_train = grp.transform(lambda s: (~s).any()) # Search if grp df has at least one row where is_test = False -> Train row
    has_test = grp.transform(lambda s: s.any())     # Search if grp has at least one row where is_test = True -> Test row
    usable = has_train & has_test                   # Indicates if the user has at least one train and test row
    df_pos = df_pos.loc[usable].copy()              # Copys all data that has at least one train and test row
    print(f"\nData shape after transform:{df_pos.shape}")
    print(f"Data after transform:\n{df_pos.head(10)}")

    # Split data in train and test data
    train_df = df_pos.loc[~df_pos["is_test"]]
    test_df = df_pos.loc[df_pos["is_test"]]

    # Build mappings 
    user_uniques = df_pos["userId"].unique()
    item_uniques = df_pos["movieId"].unique()
    user_id_to_index = {u: i for i, u in enumerate(user_uniques)}
    item_id_to_index = {m: i for i, m in enumerate(item_uniques)}
    n_users, n_items = len(user_uniques), len(item_uniques)

    # Build the coo matrices for train and test
    train_coo = build_binary_coo(train_df, user_id_to_index, item_id_to_index, n_users, n_items)
    test_coo  = build_binary_coo(test_df, user_id_to_index, item_id_to_index, n_users, n_items)

    print(f"\nShape of train matrix: {train_coo.shape}")
    print(f"Train matrix:\n{train_coo.toarray()}")

    print(f"Shape of test matrix: {test_coo.shape}")
    print(f"Test matrix:\n{test_coo.toarray()}") 

    # Store mappings
    mappings = Mappings(
        user_index_to_id=dict(enumerate(user_uniques)),
        item_index_to_id=dict(enumerate(item_uniques)),
        user_id_to_index=user_id_to_index,
        item_id_to_index=item_id_to_index,
    )

    return train_coo.tocsr(), test_coo.tocsr(), mappings


def evaluate_als(
        model: AlternatingLeastSquares,
        train_coo: csr_matrix,
        test_coo: csr_matrix,
        K: int = 10,
        num_threads: int = 0,
        show_progress: bool = False
        ) -> ALS_Metrics:
    """
    Evaluate ALS with precision@K and MAP@K using implicit.evaluation.
    Expects:
      - model: trained AlternatingLeastSquares
      - train_coo: USER×ITEM sparse matrix (COO) for train (used to filter seen items)
      - test_coo:  USER×ITEM sparse matrix (COO) for test (ground-truth relevance)
    """
    train_user_item = train_coo.tocsr()
    test_user_item  = test_coo.tocsr()

    if test_user_item.nnz == 0:
        raise ValueError("Test matrix is empty (nnz=0). Use train_percentage < 1.0 or leave-k-out for evaluation.")

    print("Evalate metrics...")
    prec = precision_at_k(
        model, train_user_item, test_user_item,
        K=K, num_threads=num_threads, show_progress=show_progress
    )
    map = mean_average_precision_at_k(
        model, train_user_item, test_user_item,
        K=K, num_threads=num_threads, show_progress=show_progress
    )
    als_metrics = ALS_Metrics(
        prec_at_k=prec,
        map_at_k=map
    )

    return als_metrics



def als_grid_search(
    train_csr: csr_matrix,
    test_csr: csr_matrix,
    bm25_K1_list: Sequence[int] = (100, 200),
    bm25_B_list: Sequence[float]  = (0.8, 1.0),
    factors_list: Sequence[int] = (128, 256),
    reg_list: Sequence[float] = (0.10, 0.20),
    iters_list: Sequence[int] = (25,),
    K: int = 10
) -> Tuple[
    Optional[AlternatingLeastSquares],
    List[ALS_Metrics],
    List[Dict[str, Any]],
    Optional[int],
]:
    '''
    Does a ASL grid search by the given parameters (bm25_K1_list, bm25_B_list, factors_list, reg_list,
    iters_list) based on maximizing the metric "map_@_k". Returns the best model, all metrics 
    (precision_@_k, map_@_k) and all parameters and the index of the best parameter combination (best_idx).
    '''
    # List to store train params and metrics
    parameter_ls = []
    metrics_ls = []

    # Points to the current index
    best_model = None
    best_idx = None

    best_score = -np.inf
    combo_iter = product(bm25_K1_list, bm25_B_list, factors_list, reg_list, iters_list)

    for idx, (K1, B, factors, reg, iters) in enumerate(combo_iter):
        print(f"\n=== Trying: BM25(K1={K1}, B={B}), ALS(factors={factors}, reg={reg}, iters={iters}) ===")

        # Compute weighted data
        train_weighted_csr = bm25_weight(train_csr, K1= K1, B=B).tocsr()

        # Fit ALS model
        model = AlternatingLeastSquares(
            factors=factors,
            regularization=reg,
            iterations=iters,
        )
        print("\nTrain model...")
        model.fit(train_weighted_csr)

        # Evaluate model results
        metrics = evaluate_als(model, train_weighted_csr, test_csr, K=K, num_threads=0, show_progress=True)
        
        # Store metrics
        metrics_ls.append(metrics)
        
        # Store parameter
        parameter_ls.append({
            "bm25_K1": K1, "bm25_B": B,
            "factors": factors, "reg": reg, "iters": iters,            
        })

        # Find best metrics -> best params
        if metrics.map_at_k > best_score:
            best_score = metrics.map_at_k
            best_model = model
            best_idx = idx                    

    return best_model, metrics_ls, parameter_ls, best_idx



def build_movie_id_dict(movies_df: pd.DataFrame) -> Dict[int, str]:
    '''
    Gets the pandas df containing the ratings and creates a dict for mapping the 
    movie_ids to the actual movie names
    '''
    # Build a quick lookup dict: {movieId: title}
    movie_id_dict = dict(zip(movies_df["movieId"], movies_df["title"]))
    return movie_id_dict


def get_movie_names(movie_id_dict: dict, movie_ids: list[int]) -> List[str]:
    '''
    Given a list of movies ids (movie_ids) it searches for the corresponding movie names 
    in the "movie_id_dict" dict and returns the movie names.
    '''
    movie_names = [movie_id_dict.get(mov_id, f"Unknown {mov_id}") for mov_id in movie_ids] 
    return movie_names


def recommend_item(
    als_model: AlternatingLeastSquares,         # Trained ALS model
    data_csr: csr_matrix,                       # BM25-weighted FULL matrix you trained the final model on
    user_id: int,                               # User id to predict the movies for
    mappings: Mappings,                         # Mappings for df ids -> csr matrix ids
    n_movies_to_rec: int = 5,                   # Number of movies to recommend
    new_user_interactions: Optional[Sequence[int]] = None,  # list of original movieIds (optional, for unknown users)
    popular_item_ids: Optional[Sequence[int]] = None,       # list of original movieIds for cold-start fallback (optional)
) -> List[int]:
    '''
    Recommends the top "n_movies_to_rec" movies for the given user id. Returns the movie ids of
    the recommended movies as a list. 
    '''
    # Helper: map internal item indices -> original ids
    def map_items_back(item_idxs):
        return [mappings.item_index_to_id[i] for i in item_idxs]

    # 1) Try known user
    if user_id in mappings.user_id_to_index:
        user_idx = mappings.user_id_to_index[user_id]
        user_row = data_csr[user_idx]

        # if user has interactions in FULL → fast path
        if user_row.nnz > 0:
            rec_items, rec_scores = als_model.recommend(
                user_idx, user_row,
                N=n_movies_to_rec,
                filter_already_liked_items=True,
                recalculate_user=False
            )
            return map_items_back(rec_items)

        # known user but no interactions in FULL → treat as cold-start below
        # (can happen if filtering removed all positives)
        # fall through to cold-start handling

    # 2) Unknown user OR known-without-interactions
    #    If we have a few interactions now, fold-in on the fly
    if new_user_interactions:
        # Build a 1×num_items temporary row with 1s at interacted items
        cols = [mappings.item_id_to_index[i] for i in new_user_interactions if i in mappings.item_id_to_index]
        if len(cols) > 0:
            data = [1.0] * len(cols)
            tmp_row = coo_matrix(
                (data, ([0]*len(cols), cols)),
                shape=(1, data_csr.shape[1]),
                dtype=data_csr.dtype
            ).tocsr()

            rec_items, rec_scores = als_model.recommend(
                0, tmp_row,
                N=n_movies_to_rec,
                filter_already_liked_items=True,
                recalculate_user=True
            )
            return map_items_back(rec_items)
    # Just return n elements from the list that contains popular movies if the new user hasn't rated any movies 
    # so far
    else:
        return popular_item_ids[:n_movies_to_rec]

    # Last resort: empty list (don’t crash pipeline)
    return []
