# =============================================================================
# MODULE 2 | LAB 2.4
# File: 04_evaluation_comparison.py
# Purpose: Head-to-head evaluation of all three models
#          Metrics: Precision@K, NDCG@K, Coverage
#          Build comparison table + 20/80 A/B user split
# Saras AI Institute | Build Predictive Models & Modern Recommenders
# =============================================================================
#
# TEACHING NOTE:
# We now have three recommenders built across Labs 2.1, 2.2, and 2.3.
# This lab evaluates them fairly on the SAME test users using THREE metrics:
#
# Precision@K : Of the top K recommendations, how many did the user buy?
# NDCG@K      : Did relevant items appear at the TOP of the list?
#               (Normalized Discounted Cumulative Gain — position matters)
# Coverage    : What fraction of the total catalog can each model recommend?
#               (A model recommending only popular items has low coverage)
#
# Finally: we implement the 20/80 A/B split — the foundation of Module 3.
# =============================================================================

import os
import pickle
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

from sklearn.preprocessing import normalize

warnings.filterwarnings('ignore')

print("=" * 60)
print("  MODULE 2 | LAB 2.4")
print("  Head-to-Head Model Evaluation")
print("  Metrics: Precision@K, NDCG@K, Coverage")
print("=" * 60)


# =============================================================================
# CONFIGURATION
# =============================================================================
K = 10

# A smaller user sample and candidate pool keep the lab executable in Colab.
# All three models use exactly the same evaluation users and candidate items.
EVAL_USER_SAMPLE = 100
CANDIDATE_POOL_SIZE = 5000
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def load_pickle_checked(path):
    """Load a pickle file with clearer errors for missing/corrupted artifacts."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required artifact not found: {path}\n"
            "Please run the prerequisite lab first."
        )

    if os.path.getsize(path) == 0:
        raise ValueError(
            f"Artifact is empty: {path}\n"
            "Please rerun the prior lab to recreate it."
        )

    try:
        with open(path, "rb") as file:
            return pickle.load(file)
    except EOFError as error:
        raise ValueError(
            f"Artifact is incomplete or corrupted: {path}\n"
            "Please rerun the prerequisite lab."
        ) from error


def ndcg_at_k(recommended_items, relevant_items, k=10):
    """
    Compute NDCG@K for one user.

    recommended_items: ordered list of recommended item IDs
    relevant_items: set of items the user actually purchased in the test period
    """
    recommended_k = recommended_items[:k]

    dcg = 0.0
    for position, item_id in enumerate(recommended_k):
        if item_id in relevant_items:
            dcg += 1.0 / np.log2(position + 2)

    ideal_hits = min(len(relevant_items), k)
    idcg = sum(1.0 / np.log2(position + 2) for position in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def compute_coverage(all_recommendations, total_catalog_size):
    """
    Coverage = unique recommended catalog items / total catalog items.
    """
    unique_recommended_items = set()

    for recommendations in all_recommendations:
        unique_recommended_items.update(recommendations)

    return (
        len(unique_recommended_items) / total_catalog_size
        if total_catalog_size > 0
        else 0.0
    )


def precision_at_k(recommended_items, relevant_items, k=10):
    """
    Compute Precision@K for one user.
    """
    if not recommended_items:
        return 0.0

    hits = len(set(recommended_items[:k]) & relevant_items)
    return hits / k


# ---------------------------------------------------------------------------
# SECTION 1: Load All Artifacts
# ---------------------------------------------------------------------------
print("\n[1] Loading all model artifacts...")

cb_artifacts = load_pickle_checked("data/cb_artifacts.pkl")
cf_artifacts = load_pickle_checked("data/cf_artifacts.pkl")
lfm_artifacts = load_pickle_checked("data/lightfm_artifacts.pkl")

# CB artifacts
cb_tfidf_matrix = cb_artifacts['tfidf_matrix']
cb_item_ids = np.asarray(cb_artifacts['item_ids'])
cb_item_to_idx = {
    item_id: idx
    for idx, item_id in enumerate(cb_item_ids)
}

# CF artifacts
cf_user_item = cf_artifacts['user_item_matrix']
cf_item_ids = np.asarray(cf_artifacts['item_ids'])
cf_item_to_idx = cf_artifacts['item_to_idx']
cf_user_to_idx = cf_artifacts['user_to_idx']

# LightFM artifacts
lfm_model = lfm_artifacts['model_hybrid']
lfm_dataset = lfm_artifacts['dataset']
lfm_train_matrix = lfm_artifacts['train_matrix']
lfm_item_features = lfm_artifacts['item_features_matrix']

# Load source event data
events = pd.read_csv("data/events.csv")
events['datetime'] = pd.to_datetime(events['timestamp'], unit='ms')

purchases = (
    events.loc[
        events['event'] == 'transaction',
        ['visitorid', 'itemid', 'timestamp', 'datetime']
    ]
    .sort_values(['visitorid', 'timestamp'])
    .copy()
)

purchases.columns = ['user_id', 'item_id', 'timestamp', 'datetime']

print("    All artifacts loaded successfully")
print(f"    CB items indexed    : {len(cb_item_ids):,}")
print(f"    CF matrix shape     : {cf_user_item.shape}")
print(f"    LightFM train matrix: {lfm_train_matrix.shape}")


# ---------------------------------------------------------------------------
# SECTION 2: Prepare Item Indexes and Candidate Catalog
# ---------------------------------------------------------------------------
print("\n[2] Preparing common candidate catalog...")

# Use only items represented in every model. This gives all models a fair
# opportunity to rank the same candidate items.
lfm_user_map, _, lfm_item_map, _ = lfm_dataset.mapping()
lfm_item_ids = np.asarray(list(lfm_item_map.keys()))
lfm_item_to_idx = lfm_item_map

common_catalog = np.intersect1d(
    np.intersect1d(cb_item_ids, cf_item_ids),
    lfm_item_ids
)

if len(common_catalog) == 0:
    raise ValueError(
        "No common items exist across the Content-Based, CF, and LightFM catalogs."
    )

rng = np.random.default_rng(RANDOM_STATE)

candidate_pool = rng.choice(
    common_catalog,
    size=min(CANDIDATE_POOL_SIZE, len(common_catalog)),
    replace=False
)

print(f"    Shared catalog items : {len(common_catalog):,}")
print(f"    Candidate pool size  : {len(candidate_pool):,}")


# ---------------------------------------------------------------------------
# SECTION 3: Build Fair Temporal Evaluation Cases
# ---------------------------------------------------------------------------
print("\n[3] Creating temporal evaluation users...")

cutoff_date = events['datetime'].quantile(0.80)

# Training purchases are historical purchases on/before the cutoff.
train_purchases = purchases[
    purchases['datetime'] <= cutoff_date
].copy()

# Test purchases are future purchases after the cutoff.
test_purchases = purchases[
    purchases['datetime'] > cutoff_date
].copy()

# A user is eligible for head-to-head evaluation if:
# 1. They have a known historical purchase to use as an item-item query.
# 2. They have at least one later purchase after the cutoff.
train_purchase_users = set(train_purchases['user_id'].unique())
test_purchase_users = set(test_purchases['user_id'].unique())

eligible_users = sorted(train_purchase_users & test_purchase_users)

if len(eligible_users) == 0:
    raise ValueError(
        "No eligible returning users were found with both historical and future purchases."
    )

eval_users = rng.choice(
    eligible_users,
    size=min(EVAL_USER_SAMPLE, len(eligible_users)),
    replace=False
)

# User -> latest historical purchase. This is the query item for CB and CF.
latest_train_purchase = (
    train_purchases
    .sort_values(['user_id', 'timestamp'])
    .groupby('user_id')
    .tail(1)
    .set_index('user_id')['item_id']
    .to_dict()
)

# User -> future purchased items. These are the relevant test items.
test_relevant_items = (
    test_purchases
    .groupby('user_id')['item_id']
    .apply(lambda item_list: set(item_list))
    .to_dict()
)

# User -> all items seen in training. LightFM recommendations will exclude them.
train_seen_items = (
    events[events['datetime'] <= cutoff_date]
    .groupby('visitorid')['itemid']
    .apply(lambda item_list: set(item_list))
    .to_dict()
)

evaluation_cases = []

for user_id in eval_users:
    query_item = latest_train_purchase.get(user_id)
    relevant_items = test_relevant_items.get(user_id, set())

    # Query and relevant items must exist in the shared model catalog.
    if query_item not in common_catalog:
        continue

    relevant_items = relevant_items & set(common_catalog)

    if len(relevant_items) == 0:
        continue

    evaluation_cases.append({
        'user_id': user_id,
        'query_item': query_item,
        'relevant_items': relevant_items,
        'seen_items': train_seen_items.get(user_id, set())
    })

if len(evaluation_cases) == 0:
    raise ValueError(
        "No valid evaluation cases remained after filtering to the shared catalog."
    )

print(f"    Temporal cutoff date : {cutoff_date}")
print(f"    Eligible users       : {len(eligible_users):,}")
print(f"    Evaluation users     : {len(evaluation_cases):,}")


# ---------------------------------------------------------------------------
# SECTION 4: Precompute Candidate Matrices
# ---------------------------------------------------------------------------
print("\n[4] Preparing efficient candidate matrices...")

# Candidate index lookups in each model's native catalog.
lfm_candidate_positions = np.array(
    [lfm_item_to_idx[item_id] for item_id in candidate_pool],
    dtype=int
)

# CF candidate item profiles: items x users.
item_user_norm = normalize(cf_user_item.T.tocsr(), norm='l2', axis=1)
print("    Candidate matrices ready")


# ---------------------------------------------------------------------------
# SECTION 5: Recommendation Functions
# ---------------------------------------------------------------------------
def get_cb_recommendations(item_id, candidate_ids, top_k=10):
    """
    Retrieve content-based recommendations from the supplied candidate pool.
    """
    if item_id not in cb_item_to_idx:
        return []

    candidate_ids = np.asarray(candidate_ids)

    # Map this user's candidate IDs into the CB TF-IDF matrix.
    candidate_positions = np.array(
        [cb_item_to_idx[candidate_id] for candidate_id in candidate_ids],
        dtype=int
    )

    query_idx = cb_item_to_idx[item_id]
    query_vector = cb_tfidf_matrix[query_idx]
    candidate_matrix = cb_tfidf_matrix[candidate_positions]

    # TF-IDF vectors are L2-normalized, so dot product equals cosine similarity.
    similarities = query_vector.dot(candidate_matrix.T).toarray().flatten()

    # Exclude the source/query item from its own recommendations.
    similarities[candidate_ids == item_id] = -1

    top_indices = np.argsort(similarities)[::-1][:top_k]
    return list(candidate_ids[top_indices])


def get_cf_recommendations(item_id, candidate_ids, top_k=10):
    """
    Retrieve Item-Item CF recommendations from the supplied candidate pool.
    """
    if item_id not in cf_item_to_idx:
        return []

    candidate_ids = np.asarray(candidate_ids)

    # Map this user's candidate IDs into the CF item-user matrix.
    candidate_positions = np.array(
        [cf_item_to_idx[candidate_id] for candidate_id in candidate_ids],
        dtype=int
    )

    query_idx = cf_item_to_idx[item_id]
    query_vector = item_user_norm[query_idx]
    candidate_matrix = item_user_norm[candidate_positions]

    # Item vectors are L2-normalized, so dot product equals cosine similarity.
    similarities = query_vector.dot(candidate_matrix.T).toarray().flatten()

    # Exclude the source/query item from its own recommendations.
    similarities[candidate_ids == item_id] = -1

    top_indices = np.argsort(similarities)[::-1][:top_k]
    return list(candidate_ids[top_indices])

def get_lfm_recommendations(user_id, candidate_ids, seen_items, top_k=10):
    """
    Retrieve LightFM hybrid recommendations from the common candidate pool,
    excluding items already seen by the user in training.
    """
    if user_id not in lfm_user_map:
        return []

    user_idx = lfm_user_map[user_id]

    candidate_item_positions = np.array(
        [lfm_item_to_idx[item_id] for item_id in candidate_ids],
        dtype=int
    )

    scores = lfm_model.predict(
        user_ids=np.full(len(candidate_item_positions), user_idx),
        item_ids=candidate_item_positions,
        item_features=lfm_item_features,
        num_threads=2
    )

    candidate_array = np.asarray(candidate_ids)

    # Avoid recommending previously observed user-item interactions.
    seen_mask = np.isin(candidate_array, list(seen_items))
    scores[seen_mask] = -np.inf

    top_indices = np.argsort(scores)[::-1][:top_k]

    recommendations = [
        candidate_array[index]
        for index in top_indices
        if np.isfinite(scores[index])
    ]

    return recommendations


# ---------------------------------------------------------------------------
# SECTION 6: Run Evaluation — All Three Models
# ---------------------------------------------------------------------------
print(f"\n[5] Running head-to-head evaluation ({len(evaluation_cases)} users, K={K})...")

cb_precisions = []
cb_ndcgs = []
cb_recs_all = []

cf_precisions = []
cf_ndcgs = []
cf_recs_all = []

lfm_precisions = []
lfm_ndcgs = []
lfm_recs_all = []

for i, case in enumerate(evaluation_cases, start=1):
    user_id = case['user_id']
    query_item = case['query_item']
    relevant_items = case['relevant_items']
    seen_items = case['seen_items']

    # Ensure each user's future relevant items are included in the candidate
    # pool. This allows the metrics to test whether models rank them highly.
    user_candidates = np.unique(
        np.concatenate([
            candidate_pool,
            np.asarray(list(relevant_items))
        ])
    )

    # Content-Based
    cb_recs = get_cb_recommendations(
        query_item,
        user_candidates,
        top_k=K
    )

    cb_precisions.append(
        precision_at_k(cb_recs, relevant_items, k=K)
    )
    cb_ndcgs.append(
        ndcg_at_k(cb_recs, relevant_items, k=K)
    )
    cb_recs_all.append(cb_recs)

    # Collaborative Filtering
    cf_recs = get_cf_recommendations(
        query_item,
        user_candidates,
        top_k=K
    )

    cf_precisions.append(
        precision_at_k(cf_recs, relevant_items, k=K)
    )
    cf_ndcgs.append(
        ndcg_at_k(cf_recs, relevant_items, k=K)
    )
    cf_recs_all.append(cf_recs)

    # LightFM Hybrid
    lfm_recs = get_lfm_recommendations(
        user_id,
        user_candidates,
        seen_items,
        top_k=K
    )

    lfm_precisions.append(
        precision_at_k(lfm_recs, relevant_items, k=K)
    )
    lfm_ndcgs.append(
        ndcg_at_k(lfm_recs, relevant_items, k=K)
    )
    lfm_recs_all.append(lfm_recs)

    if i % 25 == 0 or i == len(evaluation_cases):
        print(f"    Evaluated {i}/{len(evaluation_cases)} users...")


# ---------------------------------------------------------------------------
# SECTION 7: Compile Metrics
# ---------------------------------------------------------------------------
print("\n[6] Calculating Precision@10, NDCG@10, and coverage...")

total_catalog = len(common_catalog)

results = {
    'Content-Based (TF-IDF)': {
        'precision_at_k': float(np.mean(cb_precisions)),
        'ndcg_at_k': float(np.mean(cb_ndcgs)),
        'coverage': compute_coverage(cb_recs_all, total_catalog),
        'evaluated': len(cb_precisions)
    },
    'Item-Item CF (Implicit)': {
        'precision_at_k': float(np.mean(cf_precisions)),
        'ndcg_at_k': float(np.mean(cf_ndcgs)),
        'coverage': compute_coverage(cf_recs_all, total_catalog),
        'evaluated': len(cf_precisions)
    },
    'LightFM Hybrid (WARP)': {
        'precision_at_k': float(np.mean(lfm_precisions)),
        'ndcg_at_k': float(np.mean(lfm_ndcgs)),
        'coverage': compute_coverage(lfm_recs_all, total_catalog),
        'evaluated': len(lfm_precisions)
    }
}


# ---------------------------------------------------------------------------
# SECTION 8: Print Comparison Table
# ---------------------------------------------------------------------------
print("\n[7] FULL COMPARISON TABLE (K=10):")
print(f"\n    {'Model':<30} {'P@10':>8} {'NDCG@10':>9} {'Coverage':>10} {'Users':>7}")
print(f"    {'-' * 68}")

for model_name, metrics in results.items():
    print(
        f"    {model_name:<30} "
        f"{metrics['precision_at_k']:>8.4f} "
        f"{metrics['ndcg_at_k']:>9.4f} "
        f"{metrics['coverage']:>10.4f} "
        f"{metrics['evaluated']:>7}"
    )

best_model = max(results, key=lambda model: results[model]['ndcg_at_k'])

print(f"\n    Best model by NDCG@10: {best_model}")


# ---------------------------------------------------------------------------
# SECTION 9: 20/80 A/B User Split
# ---------------------------------------------------------------------------
print("\n[8] Building 20/80 A/B user split...")

train_users = set(
    events.loc[
        events['datetime'] <= cutoff_date,
        'visitorid'
    ].unique()
)

test_users = set(
    events.loc[
        events['datetime'] > cutoff_date,
        'visitorid'
    ].unique()
)

new_users = test_users - train_users
returning_users = test_users & train_users

print(f"    Total test users : {len(test_users):,}")

if len(test_users) > 0:
    print(
        f"    New users        : {len(new_users):,}  "
        f"({len(new_users) / len(test_users) * 100:.1f}%)"
    )
    print(
        f"    Returning users  : {len(returning_users):,}  "
        f"({len(returning_users) / len(test_users) * 100:.1f}%)"
    )

new_users_list = sorted(new_users)

if len(new_users_list) > 0:
    lightfm_count = max(1, int(len(new_users_list) * 0.20))

    lightfm_users = set(
        rng.choice(
            new_users_list,
            size=lightfm_count,
            replace=False
        )
    )
else:
    lightfm_users = set()

remaining_new = new_users - lightfm_users

print("\n    ROUTING DECISIONS:")
print(f"    -> LightFM (20% of new users) : {len(lightfm_users):,} users")
print(f"    -> ALS Module 3 (returning)   : {len(returning_users):,} users")
print(f"    -> Remaining new (explore)    : {len(remaining_new):,} users")

routing_split = {
    'lightfm_users': lightfm_users,
    'als_users': returning_users,
    'remaining_new_users': remaining_new,
    'cutoff_date': cutoff_date,
    'train_users': train_users,
    'test_users': test_users,
    'evaluation_config': {
        'k': K,
        'evaluation_user_sample': EVAL_USER_SAMPLE,
        'candidate_pool_size': CANDIDATE_POOL_SIZE,
        'random_state': RANDOM_STATE
    },
    'model_results': results
}

routing_path = "data/routing_split.pkl"
temp_routing_path = "data/routing_split_temp.pkl"

with open(temp_routing_path, "wb") as file:
    pickle.dump(
        routing_split,
        file,
        protocol=pickle.HIGHEST_PROTOCOL
    )

os.replace(temp_routing_path, routing_path)

print("\n    Saved -> data/routing_split.pkl  (Module 3 loads this)")


# ---------------------------------------------------------------------------
# SECTION 10: Visualization
# ---------------------------------------------------------------------------
print("\n[9] Plotting model comparison...")

os.makedirs("output", exist_ok=True)

model_names = list(results.keys())
p_at_k_vals = [results[model]['precision_at_k'] for model in model_names]
ndcg_vals = [results[model]['ndcg_at_k'] for model in model_names]
coverage_vals = [results[model]['coverage'] for model in model_names]

colors = ['#4C72B0', '#DD8452', '#55A868']
short_names = ['CB', 'Item-Item CF', 'LightFM']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

fig.suptitle(
    "Lab 2.4: Head-to-Head Model Evaluation (K=10)\n"
    "Content-Based vs Collaborative Filtering vs LightFM Hybrid",
    fontsize=12,
    fontweight='bold'
)

for ax, values, title in zip(
    axes,
    [p_at_k_vals, ndcg_vals, coverage_vals],
    ["Precision@10", "NDCG@10", "Catalog Coverage"]
):
    bars = ax.bar(
        range(3),
        values,
        color=colors,
        edgecolor='white',
        width=0.6
    )

    ax.set_xticks(range(3))
    ax.set_xticklabels(short_names, fontsize=10)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.grid(axis='y', alpha=0.25)

    max_value = max(values) if max(values) > 0 else 1

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max_value * 0.02),
            f"{value:.4f}",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("output/04_model_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

print("    Saved -> output/04_model_comparison.png")


# ---------------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  LAB 2.4 COMPLETE — WEEK 2 DELIVERABLE SUMMARY")
print("=" * 60)

print(f"""
  MODEL COMPARISON (K={K}):
  {'Model':<30} {'P@10':>8} {'NDCG@10':>9} {'Coverage':>10}""")

for model_name, metrics in results.items():
    print(
        f"  {model_name:<30} "
        f"{metrics['precision_at_k']:>8.4f} "
        f"{metrics['ndcg_at_k']:>9.4f} "
        f"{metrics['coverage']:>10.4f}"
    )

print(f"""
  A/B ROUTING SPLIT:
  LightFM (new users, 20%)  : {len(lightfm_users):,} users
  ALS (returning, Module 3) : {len(returning_users):,} users

  Evaluation design:
  - Temporal 80/20 split
  - Same returning users evaluated across all models
  - Candidate pool size: {CANDIDATE_POOL_SIZE:,}
  - Evaluation users: {len(evaluation_cases):,}

  Week 2 Deliverable:
  [OK] Content-Based TF-IDF recommender evaluated
  [OK] Item-Item Collaborative Filtering evaluated
  [OK] LightFM Hybrid recommender evaluated
  [OK] Precision@10, NDCG@10, and Coverage calculated
  [OK] 20/80 routing split saved -> data/routing_split.pkl

  Next: Module 3 — ALS Personalization Engine for returning users
""")

print("   Move to: Module 3 -> 01_als_personalization.py")
