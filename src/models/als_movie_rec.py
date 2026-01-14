from __future__ import annotations
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple, Mapping, Iterable, Any, TypedDict, Union
from pydantic import BaseModel

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, diags
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from implicit.evaluation import train_test_split, precision_at_k, mean_average_precision_at_k

import zlib

from itertools import product

import random

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


class ALS_Metrics(BaseModel):
    '''
    Metrics to compare the ALS model performance.
    '''
    prec_at_k: float
    map_at_k: float



# _________________________________________________________________________________________________________
# ALS functionality code
# _________________________________________________________________________________________________________

def csr_fingerprint(X) -> str:
    '''Create hash of matrices adn other stuff to check if they are consistent through runs.'''
    h = 0
    for arr in (X.indptr, X.indices, X.data):
        h = zlib.crc32(arr.view(np.uint8), h)
    return f"{h & 0xffffffff:08x}"


def _set_seed(seed: int) -> None:
    '''Set see for reproduceability.'''
    np.random.seed(seed)
    random.seed(seed)


def get_popular_items(
        df: pd.DataFrame,
        top_n: int = 50,
        threshold: float = 4.0
        ) -> List[int]:
    '''
    This function will be frequently used to build a list containing the popular items.
    This popular items can the be used for the case that a new user where no preferences
    are known needs to get movie recommendations. 
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


def apply_mask_to_csr(
        csr_matrix: csr_matrix,
        mask: np.ndarray
) -> csr_matrix:
    '''
    Apply's the given mask to the given csr matrix to zero out rows were the mask
    has the value false, the other rows stay as they are.
    '''
    n_users = csr_matrix.shape[0]
    row_mask = mask.astype(int)
    D = diags(row_mask, 0, shape=(n_users, n_users), format="csr")
    test_csr_masked = D.dot(csr_matrix)

    return test_csr_masked



def prepare_data(
        df: pd.DataFrame,
        pos_threshold: float = 4.0,
        ) -> Tuple[csr_matrix, csr_matrix, csr_matrix, Mappings, np.ndarray]:
    '''
    Prepare the data by dropping nans, keep only ratings > threshold, split into train
    and test and build scr matrixes of them. The test csr matrix will be filtered. 
    All train rows that have lessthan 5 train entries and 1 test entries will be set
    to zero. The zero lines will be automatically ignored when computing the evaluation
    metrics. Returns the train, test csr filtered test csr matrices, the mappings and
    the evaluation_set_mask used for filtering the test csr matrix.

    Parameters
    ----------
    df (pd.DataFrame):
            Input interactions with columns:
            - `userId` (int): user identifier (will be cast to int64).
            - `movieId` (int): item identifier (will be cast to int64).
            - `rating` (float): explicit rating (will be cast to float32).
            - `timestamp` (int): interaction time (will be cast to int64).
    pos_threshold (float, optional):
            Minimum rating to treat an interaction as positive (kept in the dataset).
            Defaults to 4.0.

    Returns
    -------
    train_csr: csr_matrix:
        The train csr matrix (user-item) used for training the model.
    test_csr: csr_matrix:
        Binary user–item matrix for the full test split (unmasked).
    test_csr_masked: csr_matrix
        The filtered csr matrix used for evaluating the model. All train rows that have
        less than 5 train entries and 1 test entries will be set to zero. The zero lines
        will be automatically ignored when computing the evaluation metrics.
    mappings: Mappings:
        Stores the relevant mappings needed to transfer the df user/item ids to the user/item ids of 
        the csr matrices.
    evaluation_set_mask: np.ndarray:
        Boolean mask where `True` marks users with ≥5 train items and ≥1 test item.
    '''
    print(f"\nOriginal df shape: {df.shape}")
    print(f"Original df:\n{df.head(20)}")

    user_counts = df['userId'].value_counts()
    print(f"\nuser_counts of original df \n{user_counts}")
    print(f"\nNumber of users with more than 5 ratings: {(df['userId'].value_counts() > 5).sum()}")

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
    
    # Here we wanne find the rows per user with the latest timestamp -> newest data
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
    
    print(f"\nuser_counts after last df filter set:\n{df['userId'].value_counts()}")

    user_counts = df_pos['userId'].value_counts()
    print(f"\nuser_counts after last df filter\n{user_counts}")
    print(f"\nNumber of users with more than 5 ratings: {(df_pos['userId'].value_counts() > 5).sum()}")

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

    # Filter train rows > 5 intersections, test rows > 1 intersection
    # Transform coo matrices to csr matrices
    train_csr = train_coo.tocsr()
    test_csr = test_coo.tocsr()

    # Count elements in each row for train and test (We cant just iterate over because we deal with csr)
    train_counts = np.diff(train_csr.indptr)
    test_counts = np.diff(test_csr.indptr)

    # Boolean mask of eligible users: >=5 train AND >=1 test
    print(f"\nTest data entries before masking: {test_csr.nnz}")
    evaluation_set_mask = (train_counts >= 4) & (test_counts >= 1)

    # Filter evaluation test set -> Only test samples wihich fullfill evaluation_set_mask condition
    # will stay
    test_csr_masked = apply_mask_to_csr(
        csr_matrix=test_csr,
        mask=evaluation_set_mask
    )

    print(f"Test data entries after masking: {test_csr_masked.nnz}")
    # ratio = evaluation_set_mask.sum() / train_coo.shape[0] * 100
    # print(f"\nTHe evaluation set includes {ratio:.2f} % of the original train/test data.")

    return train_csr, test_csr, test_csr_masked, mappings, evaluation_set_mask


def create_test_data():
    '''
    Create testdata for testing als model and preprocessing.
    '''
    n_users = 6
    # users_ids = np.arange(n_users)
    # movie_ids = np.arange(n_users)

    ratings = np.array([
    # User 1: 6/10 Ratings (60%)
    [4.0, 3.0, 0.0, 5.0, 0.0, 2.0, 4.0, 0.0, 3.0, 0.0],    
    
    # User 2: 7/10 Ratings (70%)  
    [0.0, 2.0, 4.0, 0.0, 3.0, 5.0, 0.0, 4.0, 0.0, 3.0],

    # User 3: 6/10 Ratings (60%)
    [3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 2.0, 5.0, 4.0, 0.0],
    
    # User 4: 7/10 Ratings (70%)
    [0.0, 0.0, 5.0, 0.0, 4.0, 3.0, 1.0, 0.0, 2.0, 4.0],
    
    # User 5: 6/10 Ratings (60%)
    [5.0, 4.0, 0.0, 3.0, 0.0, 0.0, 4.0, 2.0, 0.0, 5.0],
    
    # User 6: 7/10 Ratings (70%)
    [2.0, 0.0, 3.0, 0.0, 5.0, 4.0, 0.0, 3.0, 1.0, 0.0]
    ])
    
    user_ids = []
    movie_ids = []
    user_ratings = []
    timestamps = []

    for i, rating_ls in enumerate(ratings):
        for j, rate in enumerate(rating_ls):
            if rate > 0.0:
                user_ratings.append(rate)
                user_ids.append(i)
                movie_ids.append(j)
                timestamps.append(j)
    
    df = pd.DataFrame({
        "userId": user_ids,
        "movieId": movie_ids,
        "rating": user_ratings,
        "timestamp": timestamps
    })

    return df
    


def prepare_data_streamlit(
        df: pd.DataFrame,
        pos_threshold: float = 4.0,
        ) -> Tuple[csr_matrix, csr_matrix, csr_matrix, Mappings, np.ndarray]:
    '''
    Prepare the data by dropping nans, keep only ratings > threshold, split into train
    and test and build scr matrixes of them. The test csr matrix will be filtered. 
    All train rows that have lessthan 5 train entries and 1 test entries will be set
    to zero. The zero lines will be automaticallyignored when computing the evaluation
    metrics. 
    Stores all the steps in df's and stores them under /data/preprocessing_steps.

    Parameters
    ----------
    df (pd.DataFrame):
            Input interactions with columns:
            - `userId` (int): user identifier (will be cast to int64).
            - `movieId` (int): item identifier (will be cast to int64).
            - `rating` (float): explicit rating (will be cast to float32).
            - `timestamp` (int): interaction time (will be cast to int64).
    pos_threshold (float, optional):
            Minimum rating to treat an interaction as positive (kept in the dataset).
            Defaults to 4.0.
    '''
    print(f"\nOriginal df shape: {df.shape}")
    print(f"Original df:\n{df.head(20)}")

    user_counts = df['userId'].value_counts()
    print(f"\nuser_counts of original df \n{user_counts}")
    print(f"\nNumber of users with more than 5 ratings: {(df['userId'].value_counts() > 5).sum()}")

    # Clean data and convert types
    df_nans = df.dropna(subset=["userId", "movieId", "rating"])
    df["userId"] = df["userId"].astype(np.int64)
    df["movieId"] = df["movieId"].astype(np.int64)
    df["rating"] = df["rating"].astype(np.float32)
    df["timestamp"] = df["timestamp"].astype(np.int64)

    # Display df infos
    print(f"\nShape after drop nans: {df_nans.shape}")
    print(f"Data after drop nans:\n{df_nans.head(10)}")

    # Keep only pos ratings
    df_pos = df_nans.loc[df["rating"] >= pos_threshold].copy()
    if df_pos.empty:
        raise ValueError(f"No positives after binarization; lower pos_threshold < {pos_threshold} or data checking needed.")
    print(f"\nData after filtering by threshold {pos_threshold}: {df_pos.shape}")
    print(f"Data after filtering by threshold {pos_threshold}:\n{df_pos.head(10)}")
    
    # Here we wanne find the rows per user with the latest timestamp -> newest data
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
    
    print(f"\nuser_counts after last df filter set:\n{df['userId'].value_counts()}")

    user_counts = df_pos['userId'].value_counts()
    print(f"\nuser_counts after last df filter\n{user_counts}")
    print(f"\nNumber of users with more than 5 ratings: {(df_pos['userId'].value_counts() > 5).sum()}")

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

    # Filter train rows > 5 intersections, test rows > 1 intersection
    # Transform coo matrices to csr matrices
    train_csr = train_coo.tocsr()
    test_csr = test_coo.tocsr()

    # Count elements in each row for train and test (We cant just iterate over because we deal with csr)
    train_counts = np.diff(train_csr.indptr)
    test_counts = np.diff(test_csr.indptr)

    # Boolean mask of eligible users: >=5 train AND >=1 test
    print(f"\nTest data entries before masking: {test_csr.nnz}")
    evaluation_set_mask = (train_counts >= 4) & (test_counts >= 1)

    # Filter evaluation test set -> Only test samples wihich fullfill evaluation_set_mask condition
    # will stay
    test_csr_masked = apply_mask_to_csr(
        csr_matrix=test_csr,
        mask=evaluation_set_mask
    )

    print(f"Test data entries after masking: {test_csr_masked.nnz}")
    # ratio = evaluation_set_mask.sum() / train_coo.shape[0] * 100
    # print(f"\nTHe evaluation set includes {ratio:.2f} % of the original train/test data.")

    test_filtered_df = pd.DataFrame.sparse.from_spmatrix(test_csr_masked)
    train_csr_df = pd.DataFrame.sparse.from_spmatrix(train_csr)
    test_csr_df = pd.DataFrame.sparse.from_spmatrix(test_csr)

    # Store df ins csv files
    df.to_csv("data/preprocessing_steps/original.csv", index=False)
    df_nans.to_csv("data/preprocessing_steps/nans.csv", index=False)
    df_pos.to_csv("data/preprocessing_steps/df_pos.csv", index=False)
    train_df.to_csv("data/preprocessing_steps/train_df.csv", index=False)
    test_df.to_csv("data/preprocessing_steps/test_df.csv", index=False)
    train_csr_df.to_csv("data/preprocessing_steps/train_csr.csv", index=False)
    test_csr_df.to_csv("data/preprocessing_steps/test_csr.csv", index=False)
    test_filtered_df.to_csv("data/preprocessing_steps/test_filtered.csv", index=False)


def evaluate_als(
        model: AlternatingLeastSquares,
        train_coo: Union[csr_matrix, coo_matrix],
        test_coo: Union[csr_matrix, coo_matrix],
        K: int = 10,
        num_threads: int = 0,
        show_progress: bool = False
        ) -> ALS_Metrics:
    '''
    Evaluates a trained Alternating Least Squares (ALS) model using Precision@K
    and Mean Average Precision@K (MAP@K) metrics from `implicit.evaluation`.

    Parameters
    ----------
    model : AlternatingLeastSquares
        The trained ALS model to evaluate.
    train_coo : Union[csr_matrix, coo_matrix]
        User–item training matrix for evaluation.
    test_coo : Union[csr_matrix, coo_matrix]
        User–item test matrix that contains the ground-truth itemsthat should be 
        recommended to users.
    K : int, optional
        Cut-off rank for computing Precision@K and MAP@K. Default is 10.
    num_threads : int, optional
        Number of CPU threads to use during evaluation. Default is 0 (let the
        library decide).
    show_progress : bool, optional
        If True, displays a progress bar during metric computation. Default is False.

    Returns
    -------
    ALS_Metrics
        A dataclass instance containing the following evaluation metrics:
        - `prec_at_k`: Precision@K of the ALS model.
        - `map_at_k`:  Mean Average Precision@K of the ALS model.
    '''
    train_user_item = train_coo.tocsr()
    test_user_item  = test_coo.tocsr()

    print(f"\nTest data entries after masking in evaluation: {test_user_item.nnz}")

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
    K: int = 10,
    n_samples: int = 0,
    alpha_list: Sequence[float] = (1.0,),
    num_threads: int = 0
) -> Tuple[
    Optional[AlternatingLeastSquares],
    List[ALS_Metrics],
    List[Dict[str, Any]],
    Optional[int],
    Dict[int, float]
]:
    '''
    Performs a grid search over BM25 and ALS hyperparameters to find the best‐performing
    ALS model based on the "map_@_k" metric.

    If n_samples > 0 then only a radnomly sampled part of all aprameter combinations are
    used to train the model. This part is optinal and can be done to speed up the training
    process.

    For each parameter combination, the training matrix is BM25-weighted (and optionally
    scaled by `alpha`), an ALS model is fitted, and evaluation metrics are computed.
    Returns the best model, all metric objects, the corresponding parameter sets,
    the index of the best combination, and a record of all parameter configurations.

    Parameters
    ----------
    train_csr : csr_matrix
        User–item training matrix used to fit ALS model.
    test_csr : csr_matrix
        User–item test matrix used to evaluate each model configuration.
    bm25_K1_list : Sequence[int], optional
        List of BM25 K1 parameters controlling term saturation. Default is (100, 200).
    bm25_B_list : Sequence[float], optional
        List of BM25 B parameters controlling document-length normalization. Default is (0.8, 1.0).
    factors_list : Sequence[int], optional
        Number of latent factors (embedding dimensions) to test for ALS. Default is (128, 256).
    reg_list : Sequence[float], optional
        Regularization strengths to test. Default is (0.10, 0.20).
    iters_list : Sequence[int], optional
        Number of ALS training iterations to perform. Default is (25,).
    K : int, optional
        Cut-off rank K for computing Precision@K and MAP@K. Default is 10.
    n_samples : int, optional
        If > 0, randomly samples that many parameter combinations from the full grid 
        instead of using all. Set zero for full grid.
    alpha_list : Sequence[float], optional
        Optional multiplicative weighting factors applied to the BM25-weighted matrix.
        Default is (1.0,).
    num_threads : int, optional
        Number of CPU threads to use during ALS fitting. Default is 0 (let the library decide).

    Returns
    -------
    best_model : AlternatingLeastSquares or None
        The trained ALS model achieving the highest MAP@K score, or None if no model was trained.
    metrics_ls : list of ALS_Metrics
        List of metric objects, each containing Precision@K and MAP@K for one parameter set.
    parameter_ls : list of dict
        List of dictionaries describing the parameters corresponding to each metrics entry.
    best_idx : int or None
        Index of the best parameter combination within `parameter_ls`, or None if no model was trained.
    actual_params : dict[int, float]
        Dictionary logging all parameter combinations tried during the grid search.
    '''
    actual_params = []

    # List to store train params and metrics
    parameter_ls = []
    metrics_ls = []

    # Points to the current index
    best_model = None
    best_idx = None

    best_score = -np.inf

    # Create grid param comb
    combo_iter = list(
        product(bm25_K1_list, bm25_B_list, factors_list, reg_list, iters_list, alpha_list)
    )

    # Sample from the grid params comb
    if n_samples > 0:
        n_samples = min(n_samples, len(combo_iter))
        sampled_combo_iter = random.sample(combo_iter, n_samples)
    else:
        sampled_combo_iter = combo_iter

    # inside als_grid_search(...)
    RUN_SEED = 123456  # or derive from your data snapshot for reproducible-by-snapshot runs
    _set_seed(RUN_SEED)

    print(f"\nTest data entries after masking in grid search: {test_csr.nnz}")

    for idx, (K1, B, factors, reg, iters, alpha) in enumerate(sampled_combo_iter):
        print(f"\n=== Trying: BM25(K1={K1}, B={B}, alpha={alpha}), ALS(factors={factors}, reg={reg}, iters={iters}) ===")
        actual_params.append(
            {
                "K1": K1,
                "B": B,
                "alpha": alpha,
                "factors":factors,
                "reg": reg,
                "iters": iters
            }
        )

        # Compute weighted data
        train_weighted_csr = bm25_weight(train_csr, K1= K1, B=B).tocsr()
        # print(f"\nFingerprint of train_weighted_csr: {csr_fingerprint(train_weighted_csr)}")
        if alpha != 1.0:
            train_weighted_csr = train_weighted_csr * float(alpha)

        # Fit ALS model
        model = AlternatingLeastSquares(
            factors=factors,
            regularization=reg,
            iterations=iters,
            num_threads=num_threads,
        )
        print("\nTrain model...")
        model.fit(train_weighted_csr)

        # Evaluate model results
        metrics = evaluate_als(model, train_weighted_csr, test_csr, K=K, num_threads=num_threads, show_progress=True)
        print(f"prec_@_k= {metrics.prec_at_k}")
        print(f"map_@_k= {metrics.map_at_k}")
        
        # Store metrics
        metrics_ls.append(metrics)
        
        # Store parameter
        parameter_ls.append({
            "bm25_K1": K1, "bm25_B": B,
            "factors": factors, "reg": reg, "iters": iters,
            "alpha": alpha
        })

        # Find best metrics -> best params
        if metrics.map_at_k > best_score:
            best_score = metrics.map_at_k
            best_model = model
            best_idx = idx                    

    return best_model, metrics_ls, parameter_ls, best_idx, actual_params



def grid_search_advanced(
    train_csr: csr_matrix,
    test_csr: csr_matrix,
    bm25_K1_list: Sequence[int] = (100, 200),
    bm25_B_list: Sequence[float]  = (0.8, 1.0),
    factors_list: Sequence[int] = (128, 256),
    reg_list: Sequence[float] = (0.10, 0.20),
    iters_list: Sequence[int] = (25,),
    K: int = 10,
    num_threads: int = 0,
    show_progress: bool = True,
    n_samples: int = 12,
    f_finetune_perc: Sequence[float] = (0.85, 1.0, 1.15),
    r_finetune_perc: Sequence[float] = (0.75, 1.0, 1.25),
    alpha_list: Sequence[float] = (1.0, 5.0, 10.0, 20.0, 40.0)
) -> Tuple[
    Optional[AlternatingLeastSquares],
    List[ALS_Metrics],
    List[Dict[str, Any]],
    Optional[int],
    Dict[int, float]
]:
    '''
    Performs a three staged grid search for finding the best ALS model.

    First stage:
        Find best K1 and B1 values for a fixed baseline parameter set.
    
    Second stage:
        Using the obtained best K1 and B1 weighting values do a big search 
        to find the parameters near the optimum.

    Third stage:
        Use the best found parameter so far and do a fine grid search, to 
        optimize the model performance.
    
    Parameters
    ----------
    train_csr : csr_matrix
        User–item training matrix used to fit ALS model.
    test_csr : csr_matrix
        User–item test matrix used to evaluate each model configuration.
    bm25_K1_list : Sequence[int], optional
        List of BM25 `K1` parameters for Stage 1.
    bm25_B_list : Sequence[float], optional
        List of BM25 `B` parameters for Stage 1.
    factors_list : Sequence[int], optional
        List of ALS latent factor counts to explore in Stage 2.
    reg_list : Sequence[float], optional
        List of ALS regularization values to explore in Stage 2.
    iters_list : Sequence[int], optional
        List of ALS iteration counts to explore in Stage 2.
    K : int, optional
        Rank cut-off for computing Precision@K and MAP@K metrics.
    num_threads : int, optional
        Number of CPU threads to use for ALS training.
    show_progress : bool, optional
        If True, prints progress information for each grid search stage. 
    n_samples : int, optional
        If > 0, randomly samples that many parameter combinations during Stage 2 instead
        of testing the full grid.
    f_finetune_perc : Sequence[float], optional
        Multiplicative factors applied to the best Stage 2 `factors` value for fine-tuning.
    r_finetune_perc : Sequence[float], optional
        Multiplicative factors applied to the best Stage 2 `reg` value for fine-tuning.
    alpha_list : Sequence[float], optional
        List of `alpha` weighting values (post-BM25 scaling) used in Stage 2.
    
    Returns
    -------
    stage_3_model : AlternatingLeastSquares or None
        The final fine-tuned ALS model from Stage 3 that achieved the highest MAP@K score.
    stage_3_metr : list of ALS_Metrics
        Evaluation metrics (Precision@K and MAP@K) for all Stage 3 configurations.
    stage_3_param : list of dict
        List of parameter dictionaries corresponding to each Stage 3 metric result.
    stage_3_best_idx : int or None
        Index of the best parameter combination within `stage_3_param`.
    all_used_params : list of dict
        Aggregated list of all parameter combinations tested across all three stages.
    '''
    # List of dict containing actual param used
    all_used_params = []

    # Stage 1: Find K1, B1 starting point
    baseline = {
        "factors": 160,
        "reg": 0.1,
        "iters": 25
    }

    print(f"\nTest data entries after masking: {test_csr.nnz}")

    # Grid search with varying K1, B1 values and fixed base line 
    # Do grid search 
    _, stage_1_metr, stage_1_param, stage_1_best_idx, actual_params = als_grid_search(
        train_csr=train_csr, 
        test_csr=test_csr,
        bm25_K1_list=bm25_K1_list,
        bm25_B_list=bm25_B_list,
        factors_list=[baseline["factors"]],
        reg_list=[baseline["reg"]],
        iters_list=[baseline["iters"]],
        K=K,
        alpha_list=(1.0,),
        num_threads=num_threads
    )
    all_used_params += actual_params

    # Stage 2: Big area search with fixed K1, B1
    # Do grid search 
    _, stage_2_metr, stage_2_param, stage_2_best_idx, actual_params = als_grid_search(
        train_csr=train_csr, 
        test_csr=test_csr,
        bm25_K1_list=[stage_1_param[stage_1_best_idx]["bm25_K1"]],
        bm25_B_list=[stage_1_param[stage_1_best_idx]["bm25_B"]],
        factors_list=factors_list,
        reg_list=reg_list,
        iters_list=iters_list,
        K=K,
        n_samples=n_samples,
        alpha_list=alpha_list,
        num_threads=num_threads
    )
    all_used_params += actual_params

    # Stage 3: Fine search near best parameter of stage 2
    # Define safety factor and reg vals
    stage_3_factors = [max(1, int(stage_2_param[stage_2_best_idx]["factors"] * perc)) for perc in f_finetune_perc]
    stage_3_regs = [max(1e-8, stage_2_param[stage_2_best_idx]["reg"] * perc) for perc in r_finetune_perc]
    best_alpha = stage_2_param[stage_2_best_idx]["alpha"]
    
    # Do grid search 
    stage_3_model, stage_3_metr, stage_3_param, stage_3_best_idx, actual_params = als_grid_search(
        train_csr=train_csr, 
        test_csr=test_csr,
        bm25_K1_list=[stage_2_param[stage_2_best_idx]["bm25_K1"]],
        bm25_B_list=[stage_2_param[stage_2_best_idx]["bm25_B"]],
        factors_list=stage_3_factors,
        reg_list=stage_3_regs,
        iters_list=[stage_2_param[stage_2_best_idx]["iters"]],
        K=K,
        alpha_list=(best_alpha, ),
        num_threads=num_threads
    )
    all_used_params += actual_params

    return stage_3_model, stage_3_metr, stage_3_param, stage_3_best_idx, all_used_params



def build_movie_id_dict(movies_df: pd.DataFrame) -> Dict[int, dict[str, str]]:
    '''
    Builds a lookup dictionary:
      {movieId: {"title": str, "genres": str}}
    '''
    # Build a quick lookup dict: {movieId: title}
    movie_id_dict = {
        int(row["movieId"]): {
            "title": row["title"],
            "genres": row.get("genres", "Unknown")
        }
        for _, row in movies_df.iterrows()
    }
    return movie_id_dict


def get_movie_names(movie_id_dict: dict, movie_ids: list[int]) -> List[str]:
    '''
    Given a list of movies ids (movie_ids) it searches for the corresponding movie names 
    in the "movie_id_dict" dict and returns the movie names.
    '''
    movie_names = [movie_id_dict.get(mov_id, f"Unknown {mov_id}") for mov_id in movie_ids] 
    return movie_names


def get_movie_metadata(movie_id_dict: dict, movie_ids: list[int]) -> tuple[list[str], list[str]]:
    """
    Given a list of movie IDs, returns two aligned lists:
    - movie_titles
    - movie_genres
    """
    movie_titles, movie_genres = [], []

    for mov_id in movie_ids:
        movie_entry = movie_id_dict.get(mov_id)
        if isinstance(movie_entry, dict):
            # ✅ New schema: nested dict with title + genres
            movie_titles.append(movie_entry.get("title", f"Unknown {mov_id}"))
            movie_genres.append(movie_entry.get("genres", "Unknown"))
        else:
            # ✅ Old schema: title only
            movie_titles.append(movie_entry)
            movie_genres.append("Unknown")

    return movie_titles, movie_genres


def recommend_item(
    als_model: AlternatingLeastSquares,         # Trained ALS model
    data_csr: csr_matrix,                       # BM25-weighted FULL matrix you trained the final model on
    user_id: int,                                  # User id to predict the movies for
    mappings: Mappings,                         # Mappings for df ids -> csr matrix ids
    n_movies_to_rec: int = 5,                   # Number of movies to recommend
    new_user_interactions: Optional[Sequence[int]] = None,  # list of original movieIds (optional, for unknown users)
    popular_item_ids: Optional[Sequence[int]] = None,       # list of original movieIds for cold-start fallback (optional)
) -> List[int]:
    '''
    Recommends the top-N movie IDs for a given user using the given trained ALS model.

    If the given user is not part of the matrix that represents the data, the function 
    first trys to compute new recommendation based on the new_user_interactions which
    is a list of movies he likes. 
    If this list is empty, then the user gets recommendations based on a list of popular
    items. This solves the cold start problem.

    Parameters
    ----------
    als_model : AlternatingLeastSquares
        The trained ALS model used for generating recommendations.
    data_csr : csr_matrix
        The BM25-weighted user–item matrix that the final ALS model was trained on.
        Must share the same user/item index mapping as `mappings`.
    user_id : int
        The original user identifier to recommend for (not CSR row index).
    mappings : Mappings
        Stores the relevant mappings needed to transfer the df user/item ids to the 
        user/item ids of the csr matrices.
    n_movies_to_rec : int, optional
        Number of recommendations to return. 
    new_user_interactions : Sequence[int] or None, optional
        List of mvoies the user likes.
    popular_item_ids : Sequence[int] or None, optional
        List of movies used for the cold start case. This list guarantees user recommendation.

    Returns
    -------
    recommendations: List[int]
        List of 'n_movies_to_rec' recommendations for the given user.
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


def main():
    df_ratings = create_test_data()
    prepare_data_streamlit(
        df=df_ratings,
        pos_threshold=3
    )


if __name__ == "__main__":
    main()