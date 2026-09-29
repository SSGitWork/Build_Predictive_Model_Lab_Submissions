# =============================================================================
# MODULE 3 | LAB 3.1
# File: 01_als_personalization.py
# Purpose: Build a personalized recommendation engine using Alternating
#          Least Squares (ALS) on implicit interaction confidence matrices.
# Saras AI Institute | Build Predictive Models & Modern Recommenders
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import pickle
import time
import warnings

warnings.filterwarnings('ignore')

from implicit.als import AlternatingLeastSquares
from implicit.evaluation import ndcg_at_k, precision_at_k

print("=" * 60)
print("  MODULE 3 | LAB 3.1")
print("  ALS Personalization Engine")
print("  Method: Alternating Least Squares on Implicit Feedback")
print("=" * 60)


# ---------------------------------------------------------------------------
# DEMO PARAMETERS
# ---------------------------------------------------------------------------
N_FACTORS = 64
N_ITERATIONS = 10
N_USERS = 50000
N_ITEMS = 10000
RANDOM_STATE = 42

print(
    f"\n  Demo scale: {N_USERS:,} users | {N_ITEMS:,} items | "
    f"{N_FACTORS} factors | {N_ITERATIONS} iterations\n"
)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def build_weighted_user_item_matrix(
    event_frame,
    confidence_map,
    user_mapping,
    item_mapping,
    matrix_shape
):
    """
    Build a confidence-weighted sparse user-item matrix using fixed user and
    item mappings. Fixed mappings ensure all experiments use identical matrix
    dimensions and can be compared fairly.
    """
    temp = event_frame.copy()

    temp['confidence'] = temp['event'].map(confidence_map).fillna(0.0)

    aggregated = (
        temp
        .groupby(['visitorid', 'itemid'], as_index=False)['confidence']
        .sum()
    )

    aggregated['row'] = aggregated['visitorid'].map(user_mapping)
    aggregated['col'] = aggregated['itemid'].map(item_mapping)

    aggregated = aggregated.dropna(subset=['row', 'col']).copy()

    if len(aggregated) == 0:
        return sp.csr_matrix(matrix_shape, dtype=np.float32)

    return sp.csr_matrix(
        (
            aggregated['confidence'].astype(np.float32).values,
            (
                aggregated['row'].astype(int).values,
                aggregated['col'].astype(int).values
            )
        ),
        shape=matrix_shape,
        dtype=np.float32
    )


def remove_seen_test_pairs(train_matrix, test_matrix):
    """
    Remove future test entries already observed during training.

    This makes the ranking evaluation meaningful: ALS is only rewarded for
    recommending unseen future user-item interactions.
    """
    train_coo = train_matrix.tocoo()
    test_coo = test_matrix.tocoo()

    train_pairs = set(zip(train_coo.row, train_coo.col))

    keep_mask = np.array(
        [
            (row, col) not in train_pairs
            for row, col in zip(test_coo.row, test_coo.col)
        ]
    )

    return sp.csr_matrix(
        (
            test_coo.data[keep_mask],
            (
                test_coo.row[keep_mask],
                test_coo.col[keep_mask]
            )
        ),
        shape=test_matrix.shape,
        dtype=np.float32
    )


def safe_ndcg(model, train_user_item_matrix, test_user_item_matrix, k=10):
    """
    Calculate NDCG@K safely. implicit evaluation expects matrices in
    user-by-item orientation, unlike ALS.fit(), which uses item-by-user.
    """
    if test_user_item_matrix.nnz == 0:
        return 0.0

    try:
        return ndcg_at_k(
            model,
            train_user_item_matrix,
            test_user_item_matrix,
            K=k,
            show_progress=False
        )
    except TypeError:
        # Compatibility fallback for implicit versions without show_progress.
        return ndcg_at_k(
            model,
            train_user_item_matrix,
            test_user_item_matrix,
            K=k
        )


# ---------------------------------------------------------------------------
# SECTION 1: Load Events and Routing Split
# ---------------------------------------------------------------------------
print("[1] Loading events and Module 2 routing split...")

events = pd.read_csv("data/events.csv")
events['datetime'] = pd.to_datetime(events['timestamp'], unit='ms')

with open("data/routing_split.pkl", "rb") as f:
    routing = pickle.load(f)

als_users = routing['als_users']
lfm_users = routing['lightfm_users']
cutoff_date = pd.to_datetime(routing['cutoff_date'])

print(f"    Total events       : {len(events):,}")
print(f"    ALS target users   : {len(als_users):,}")
print(f"    LightFM users      : {len(lfm_users):,}")
print(f"    Train cutoff       : {cutoff_date}")


# ---------------------------------------------------------------------------
# SECTION 2: Subsample for Demo Speed
# ---------------------------------------------------------------------------
print("\n[2] Subsampling to demo scale...")

# Only retain users routed to ALS, which are returning users from Lab 2.4.
als_events = events[events['visitorid'].isin(als_users)].copy()

# TODO: Isolate top users and items based on total interaction volume counts to speed up processing
top_users = (
    als_events['visitorid']
    .value_counts()
    .head(N_USERS)
    .index
)

events_for_top_users = als_events[
    als_events['visitorid'].isin(top_users)
].copy()

top_items = (
    events_for_top_users['itemid']
    .value_counts()
    .head(N_ITEMS)
    .index
)

# TODO: Filter the events dataframe to include only records matching top_users and top_items lists
events_sub = events_for_top_users[
    events_for_top_users['itemid'].isin(top_items)
].copy()

print(f"    Subsampled events : {len(events_sub):,}")
print(f"    Selected users    : {events_sub['visitorid'].nunique():,}")
print(f"    Selected items    : {events_sub['itemid'].nunique():,}")


# ---------------------------------------------------------------------------
# SECTION 3: Build Confidence-Weighted Interaction Matrix
# ---------------------------------------------------------------------------
print("\n[3] Building confidence-weighted interaction matrix...")

# TODO: Define mapping parameters to structure implicit signal strengths ($C_{ui} = 1 + \alpha R_{ui}$)
# Values to set: view -> 1.0, addtocart -> 5.0, transaction -> 40.0
ALPHA_VIEW = 1.0
ALPHA_ADDTOCART = 5.0
ALPHA_TXN = 40.0

conf_map = {
    'view': ALPHA_VIEW,
    'addtocart': ALPHA_ADDTOCART,
    'transaction': ALPHA_TXN
}

events_sub['confidence'] = events_sub['event'].map(conf_map).fillna(0.0)

# TODO: Generate continuous unique user and item arrays from events_sub
user_ids = events_sub['visitorid'].unique()
item_ids = events_sub['itemid'].unique()

# TODO: Build conversion dictionary lookups mapping raw identifiers to continuous coordinate integers
user_to_idx = {
    user_id: index
    for index, user_id in enumerate(user_ids)
}

item_to_idx = {
    item_id: index
    for index, item_id in enumerate(item_ids)
}

# TODO: Aggregate calculated confidence tracking weights per user-item intersection pair
# Hint: Group events_sub by ['visitorid', 'itemid'] and sum 'confidence' columns, then reset index
interactions = (
    events_sub
    .groupby(['visitorid', 'itemid'], as_index=False)['confidence']
    .sum()
)

# TODO: Map raw identifiers within interactions down to coordinate row/col index indices
row = interactions['visitorid'].map(user_to_idx).astype(int).values
col = interactions['itemid'].map(item_to_idx).astype(int).values
dat = interactions['confidence'].astype(np.float32).values

# TODO: Build the sparse matrix tracking user item shapes
# Hint: Instantiate a sp.csr_matrix using configuration tuples: (dat, (row, col))
user_item = sp.csr_matrix(
    (dat, (row, col)),
    shape=(len(user_ids), len(item_ids)),
    dtype=np.float32
)

# TODO: Transpose user_item to generate the item_user matrix explicitly as a CSR matrix
item_user = user_item.T.tocsr()

print(f"    User-item matrix shape: {user_item.shape}")
print(f"    Item-user matrix shape: {item_user.shape}")
print(f"    Non-zero interactions : {user_item.nnz:,}")


# ---------------------------------------------------------------------------
# SECTION 4: Temporal Train/Test Split
# ---------------------------------------------------------------------------
print("\n[4] Temporal train/test split...")

# TODO: Partition events_sub into training logs (<= cutoff_date) and testing logs (> cutoff_date and transaction events only)
train_events = events_sub[
    events_sub['datetime'] <= cutoff_date
].copy()

test_events = events_sub[
    (events_sub['datetime'] > cutoff_date) &
    (events_sub['event'] == 'transaction')
].copy()

# TODO: Aggregate train interactions and map them to row/col indexing arrays to build train_user_item
train_user_item = build_weighted_user_item_matrix(
    event_frame=train_events,
    confidence_map=conf_map,
    user_mapping=user_to_idx,
    item_mapping=item_to_idx,
    matrix_shape=(len(user_ids), len(item_ids))
)

# Remember to expose item_user matrix variants for implicit model ingestion
train_item_user = train_user_item.T.tocsr()

# TODO: Compile test purchases matrix using np.ones structure to build test labels
# Note: Ensure you check that test item and user IDs map within index dictionary boundaries using .notna()
test_events_mapped = test_events.copy()

test_events_mapped['row'] = test_events_mapped['visitorid'].map(user_to_idx)
test_events_mapped['col'] = test_events_mapped['itemid'].map(item_to_idx)

test_events_mapped = test_events_mapped.dropna(
    subset=['row', 'col']
).drop_duplicates(
    subset=['row', 'col']
).copy()

test_user_item = sp.csr_matrix(
    (
        np.ones(len(test_events_mapped), dtype=np.float32),
        (
            test_events_mapped['row'].astype(int).values,
            test_events_mapped['col'].astype(int).values
        )
    ),
    shape=(len(user_ids), len(item_ids)),
    dtype=np.float32
)

# Remove test interactions already seen during the training period.
test_user_item = remove_seen_test_pairs(
    train_user_item,
    test_user_item
)

test_item_user = test_user_item.T.tocsr()

print(f"    Train matrix shape : {train_user_item.shape}")
print(f"    Test matrix shape  : {test_user_item.shape}")
print(f"    Train interactions : {train_user_item.nnz:,}")
print(f"    Test purchases     : {test_user_item.nnz:,}")


# ---------------------------------------------------------------------------
# SECTION 5: Train ALS Model
# ---------------------------------------------------------------------------
print("\n[5] Training ALS model...")

# TODO: Instantiate an AlternatingLeastSquares model object.
# Parameters: factors=N_FACTORS, regularization=0.01, iterations=N_ITERATIONS, use_gpu=False, random_state=42
model = AlternatingLeastSquares(
    factors=N_FACTORS,
    regularization=0.01,
    iterations=N_ITERATIONS,
    use_gpu=False,
    random_state=RANDOM_STATE
)

# TODO: Fit the model configuration using the train_item_user matrix
t0 = time.time()
model.fit(train_user_item, show_progress=True)
train_time = time.time() - t0

print(f"\n    Training complete in {train_time:.1f}s")

print(f"    User factors shape : {model.user_factors.shape}")
print(f"    Item factors shape : {model.item_factors.shape}")

# ---------------------------------------------------------------------------
# SECTION 6: Evaluate — NDCG@10
# ---------------------------------------------------------------------------
print("\n[6] Evaluating ALS model...")

# TODO: Calculate NDCG@10 rankings using implicit evaluation tools
# Hint: Call ndcg_at_k() providing model, train_item_user, test_item_user, and K=10
baseline_ndcg = safe_ndcg(
    model,
    train_user_item,
    test_user_item,
    k=10
)

print(f"    ALS NDCG@10    : {baseline_ndcg:.4f}")


# ---------------------------------------------------------------------------
# SECTION 7: Confidence Weight Experiment
# ---------------------------------------------------------------------------
print("\n[7] Confidence weight tuning experiment...")

configs = [
    ("Low emphasis  (view=1, cart=2, txn=10)", 1.0, 2.0, 10.0),
    ("Balanced      (view=1, cart=5, txn=40)", 1.0, 5.0, 40.0),
    ("Purchase-heavy(view=0.5,cart=3, txn=80)", 0.5, 3.0, 80.0),
]

print(f"\n    {'Config':<45} {'NDCG@10':>8} {'Time(s)':>8}")
print(f"    {'-' * 63}")

best_ndcg = -1
best_model = model
best_config = "Balanced (view=1, cart=5, txn=40)"
best_item_user = train_item_user
best_user_item = train_user_item

config_names = []
config_ndcgs = []
config_times = []

for config_name, a_view, a_cart, a_txn in configs:
    # TODO: Loop over the weight configurations, construct temporary confidence maps,
    # aggregate weights, build a temporary sparse matrix, fit a new ALS model,
    # and track the best performing trial based on NDCG@10 scores.

    temp_conf_map = {
        'view': a_view,
        'addtocart': a_cart,
        'transaction': a_txn
    }

    temp_train_user_item = build_weighted_user_item_matrix(
        event_frame=train_events,
        confidence_map=temp_conf_map,
        user_mapping=user_to_idx,
        item_mapping=item_to_idx,
        matrix_shape=(len(user_ids), len(item_ids))
    )

    temp_train_item_user = temp_train_user_item.T.tocsr()

    temp_model = AlternatingLeastSquares(
        factors=N_FACTORS,
        regularization=0.01,
        iterations=N_ITERATIONS,
        use_gpu=False,
        random_state=RANDOM_STATE
    )

    config_start_time = time.time()
    temp_model.fit(
        temp_train_user_item,
        show_progress=False
    )

    config_train_time = time.time() - config_start_time

    temp_ndcg = safe_ndcg(
        temp_model,
        temp_train_user_item,
        test_user_item,
        k=10
    )

    config_names.append(config_name)
    config_ndcgs.append(temp_ndcg)
    config_times.append(config_train_time)

    print(
        f"    {config_name:<45} "
        f"{temp_ndcg:>8.4f} "
        f"{config_train_time:>8.1f}"
    )

    if temp_ndcg > best_ndcg:
        best_ndcg = temp_ndcg
        best_model = temp_model
        best_config = config_name
        best_item_user = temp_train_item_user
        best_user_item = temp_train_user_item

# Reset variables to capture optimal structures found
model = best_model
item_user = best_item_user
user_item = best_user_item
train_user_item = best_user_item
train_item_user = best_item_user

print(f"\n    Best configuration: {best_config}")
print(f"    Best NDCG@10      : {best_ndcg:.4f}")


# ---------------------------------------------------------------------------
# SECTION 8: Sample Recommendations
# ---------------------------------------------------------------------------
print("\n[8] Sample recommendations for returning users...")

if user_ids is not None and model is not None:
    sample_users = user_ids[:5]

    for user_id in sample_users:
        user_idx = user_to_idx[user_id]

        # TODO: Retrieve top 5 recommendations from your trained model for user_idx
        # Hint: Use model.recommend(userid=..., user_items=train_user_item[user_idx], N=5, filter_already_liked_items=True)
        # Unpack the resulting item_indices and scores to print output values
        item_indices, scores = model.recommend(
            userid=user_idx,
            user_items=train_user_item[user_idx],
            N=5,
            filter_already_liked_items=True
        )

        recommended_item_ids = item_ids[item_indices]

        print(f"\n    User {user_id}:")
        for rank, (recommended_item, score) in enumerate(
            zip(recommended_item_ids, scores),
            start=1
        ):
            print(
                f"      {rank}. Item {recommended_item} "
                f"| score = {score:.4f}"
            )


# ---------------------------------------------------------------------------
# SECTION 9: Visualizations
# ---------------------------------------------------------------------------
print("\n[9] Generating visualizations...")

os.makedirs("output", exist_ok=True)

# --- Latent Embedding Profile Vector Magnitudes ---
# TODO: Calculate L2 norms across both user and item latent matrices (model.user_factors, model.item_factors)
# Hint: Use np.linalg.norm(..., axis=1)
user_norms = np.linalg.norm(model.user_factors, axis=1)
item_norms = np.linalg.norm(model.item_factors, axis=1)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

fig.suptitle(
    "Lab 3.1: ALS Personalization Engine\n"
    "Embedding Analysis & Confidence Weight Tuning",
    fontsize=12,
    fontweight='bold'
)

# TODO: Plot histograms on axes[0] and axes[1] detailing user and item L2 embedding norms distributions
axes[0].hist(
    user_norms,
    bins=50,
    color='#4C72B0',
    edgecolor='white'
)
axes[0].set_title("User Embedding Norm Distribution")
axes[0].set_xlabel("L2 Norm")
axes[0].set_ylabel("Number of Users")

axes[1].hist(
    item_norms,
    bins=50,
    color='#DD8452',
    edgecolor='white'
)
axes[1].set_title("Item Embedding Norm Distribution")
axes[1].set_xlabel("L2 Norm")
axes[1].set_ylabel("Number of Items")

# --- Config Performance Comparisons ---
# TODO: Construct a bar plot layout on axes[2] mapping calculated NDCG scores across configs list
# Hint: Use axes[2].bar()
short_config_names = [
    "Low\nEmphasis",
    "Balanced",
    "Purchase\nHeavy"
]

bars = axes[2].bar(
    short_config_names,
    config_ndcgs,
    color=['#4C72B0', '#55A868', '#DD8452'],
    edgecolor='white'
)

for bar, score in zip(bars, config_ndcgs):
    axes[2].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{score:.4f}",
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='bold'
    )

axes[2].set_title("Confidence Weight Tuning\nNDCG@10 per Config")
axes[2].set_ylabel("NDCG@10")
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("output/01_als_analysis.png", dpi=150, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------------------
# SECTION 10: Save Artifacts
# ---------------------------------------------------------------------------
print("\n[10] Saving ALS artifacts...")

# TODO: Export tracking matrices, mappings lists, indexes, and factors down to a binary pickle format
# Target Path: "data/als_artifacts.pkl"
als_artifacts = {
    'model': model,
    'user_ids': user_ids,
    'item_ids': item_ids,
    'user_to_idx': user_to_idx,
    'item_to_idx': item_to_idx,
    'user_item_matrix': user_item,
    'item_user_matrix': item_user,
    'train_user_item_matrix': train_user_item,
    'train_item_user_matrix': train_item_user,
    'test_user_item_matrix': test_user_item,
    'test_item_user_matrix': test_item_user,
    'user_factors': model.user_factors,
    'item_factors': model.item_factors,
    'baseline_ndcg_at_10': baseline_ndcg,
    'best_ndcg_at_10': best_ndcg,
    'best_config': best_config,
    'config_names': config_names,
    'config_ndcgs': config_ndcgs,
    'config_times': config_times,
    'parameters': {
        'n_factors': N_FACTORS,
        'n_iterations': N_ITERATIONS,
        'n_users_limit': N_USERS,
        'n_items_limit': N_ITEMS,
        'cutoff_date': cutoff_date,
        'random_state': RANDOM_STATE
    }
}

artifact_path = "data/als_artifacts.pkl"
temporary_path = "data/als_artifacts_temp.pkl"

with open(temporary_path, "wb") as f:
    pickle.dump(
        als_artifacts,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

os.replace(temporary_path, artifact_path)

print("    Saved -> data/als_artifacts.pkl")
print("    Move to: 02_faiss_index.py")
