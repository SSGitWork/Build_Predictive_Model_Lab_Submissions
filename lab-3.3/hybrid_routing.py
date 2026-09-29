# =============================================================================
# MODULE 3 | LAB 3.3
# File: 03_hybrid_routing.py
# Purpose: Build a production-grade Hybrid Routing and Post-Processing layer
#          integrating ALS (returning users), LightFM (cold-start),
#          Already-Purchased Exclusion, and MMR Diversity re-ranking.
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

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from lightfm import LightFM

print("=" * 60)
print("  MODULE 3 | LAB 3.3")
print("  Hybrid Routing Layer")
print("  LightFM (new) + ALS (returning) + MMR + Exclusion")
print("=" * 60)


# ---------------------------------------------------------------------------
# Helper Function: Load Pickles Safely
# ---------------------------------------------------------------------------
def load_pickle_checked(path):
    """Load a non-empty pickle file with helpful errors."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required artifact not found: {path}. "
            "Run the prerequisite lab first."
        )

    if os.path.getsize(path) == 0:
        raise ValueError(
            f"Artifact is empty: {path}. "
            "Rerun the corresponding prerequisite lab."
        )

    try:
        with open(path, "rb") as file:
            return pickle.load(file)
    except EOFError as error:
        raise ValueError(
            f"Artifact is incomplete or corrupted: {path}. "
            "Rerun the corresponding prerequisite lab."
        ) from error


# ---------------------------------------------------------------------------
# SECTION 1: Load All Artifacts
# ---------------------------------------------------------------------------
print("\n[1] Loading all artifacts...")

routing = load_pickle_checked("data/routing_split.pkl")
lfm_art = load_pickle_checked("data/lightfm_artifacts.pkl")
als_art = load_pickle_checked("data/als_artifacts.pkl")
faiss_art = load_pickle_checked("data/faiss_artifacts.pkl")

als_users = set(routing['als_users'])
lfm_users = set(routing['lightfm_users'])
cutoff_date = pd.to_datetime(routing['cutoff_date'])

lfm_model = lfm_art['model_hybrid']
lfm_dataset = lfm_art['dataset']
lfm_item_features = lfm_art['item_features_matrix']
lfm_train_matrix = lfm_art['train_matrix']

als_model = als_art['model']
als_user_to_idx = als_art['user_to_idx']
als_item_to_idx = als_art['item_to_idx']
als_user_ids = np.asarray(als_art['user_ids'])
als_item_ids = np.asarray(als_art['item_ids'])

# Use the training matrix if available. This contains historical confidence
# interactions used for ALS recommendation filtering.
als_user_item = als_art.get(
    'train_user_item_matrix',
    als_art.get('user_item_matrix')
)

item_factors_norm = np.asarray(faiss_art['item_factors_norm'])
user_factors_norm = np.asarray(faiss_art['user_factors_norm'])
faiss_item_ids = np.asarray(faiss_art['item_ids'])
faiss_user_to_idx = faiss_art['user_to_idx']

events = pd.read_csv("data/events.csv")
events['datetime'] = pd.to_datetime(events['timestamp'], unit='ms')

# TODO: Build a historical purchase dictionary tracking already bought items per user
# Hint: Filter events down to 'transaction', group by 'visitorid', pull 'itemid',
# and transform into a dictionary mapping user keys to sets of item IDs.
purchases = (
    events[events['event'] == 'transaction']
    .groupby('visitorid')['itemid']
    .apply(lambda item_list: set(item_list))
    .to_dict()
)

faiss_index = None

if FAISS_AVAILABLE:
    try:
        faiss_index_path = faiss_art.get(
            'faiss_index_path',
            "data/faiss_index.bin"
        )

        faiss_index = faiss.read_index(faiss_index_path)

        print(f"    FAISS index loaded : {faiss_index.ntotal:,} items")

    except Exception:
        FAISS_AVAILABLE = False
        faiss_index = None
        print("    FAISS index not found — using numpy fallback")

lfm_user_map, _, lfm_item_map, _ = lfm_dataset.mapping()
lfm_item_ids_list = list(lfm_item_map.keys())
n_lfm_items = len(lfm_item_ids_list)

faiss_id_to_idx = {
    item_id: index
    for index, item_id in enumerate(faiss_item_ids)
}

print(f"    ALS users available    : {len(als_user_to_idx):,}")
print(f"    LightFM users available: {len(lfm_user_map):,}")
print(f"    Historical buyers      : {len(purchases):,}")


# ---------------------------------------------------------------------------
# SECTION 2: Post-Processing Functions
# ---------------------------------------------------------------------------
print("\n[2] Defining post-processing functions...")


# --- Score Normalization ---
def min_max_normalize(scores):
    """
    Scale absolute engine output values to a uniform [0, 1] interval.
    Formula: S_norm = (S - S_min) / (S_max - S_min)
    """
    # TODO: Implement min-max normalization. Convert scores to an array.
    # Handle the boundary case where s_max == s_min by returning an array of 0.5s.
    scores = np.asarray(scores, dtype=np.float32)

    if len(scores) == 0:
        return np.array([], dtype=np.float32)

    score_min = scores.min()
    score_max = scores.max()

    if score_max == score_min:
        return np.full(len(scores), 0.5, dtype=np.float32)

    return (scores - score_min) / (score_max - score_min)


# --- Already-Purchased Exclusion ---
def exclude_purchased(items, scores, user_id):
    """
    Filter recommendation lists to remove items that a user has already bought.
    """
    # TODO: Verify if user_id exists in the purchases dictionary.
    # Loop over the zipped items and scores, filtering out items present in the purchase history.
    # Return two unzipped collections: (filtered_items, filtered_scores).
    if user_id not in purchases:
        return list(items), list(scores)

    previously_purchased = purchases[user_id]

    filtered_pairs = [
        (item_id, score)
        for item_id, score in zip(items, scores)
        if item_id not in previously_purchased
    ]

    if not filtered_pairs:
        return [], []

    filtered_items, filtered_scores = zip(*filtered_pairs)

    return list(filtered_items), list(filtered_scores)


# --- MMR Diversity Injection ---
def mmr_rerank(items, scores, embeddings, id_to_idx, top_k=10, lam=0.5):
    """
    Maximal Marginal Relevance greedy re-ranking optimization.

    Objective:
    argmax [lambda * relevance - (1 - lambda) * max_similarity_to_selected]
    """
    items = list(items)
    scores = list(scores)

    if len(items) == 0:
        return [], []

    # TODO: Normalize scores using your min_max_normalize tool
    norm_scores = min_max_normalize(scores)

    # TODO: Extract baseline matrices containing (item_id, norm_score, normalized_embedding_vector)
    # Ensure items exist in your id_to_idx dictionary mapping boundaries
    valid = [
        (
            item_id,
            float(norm_score),
            embeddings[id_to_idx[item_id]]
        )
        for item_id, norm_score in zip(items, norm_scores)
        if item_id in id_to_idx
    ]

    if not valid:
        return items[:top_k], scores[:top_k]

    selected_items = []
    selected_scores = []
    selected_embs = []

    remaining = list(range(len(valid)))

    for _ in range(min(top_k, len(valid))):
        if not remaining:
            break

        if not selected_embs:
            # TODO: Pick the first entry based on maximum relevance score alone
            best = max(
                remaining,
                key=lambda index: valid[index][1]
            )

        else:
            # TODO: Convert selected_embs into a numpy matrix array
            sel_matrix = np.asarray(selected_embs, dtype=np.float32)
            mmr_vals = []

            # TODO: Iterate over remaining index options, calculate relevance weights,
            # calculate similarity products against the sel_matrix, and solve the MMR optimization formula
            for i in remaining:
                rel = lam * valid[i][1]
                sims = valid[i][2] @ sel_matrix.T
                div = (1 - lam) * np.max(sims)

                mmr_vals.append(rel - div)

            best = remaining[np.argmax(mmr_vals)]

        # Append chosen targets to tracking lists
        selected_items.append(valid[best][0])
        selected_scores.append(valid[best][1])
        selected_embs.append(valid[best][2])

        remaining.remove(best)

    return selected_items, selected_scores


# ---------------------------------------------------------------------------
# SECTION 3: Engine Recommendation Functions
# ---------------------------------------------------------------------------
print("\n[3] Defining engine recommendation functions...")


def get_als_recs(user_id, top_k=30):
    """
    Query the personalized ALS embedding space using FAISS index lookups or
    NumPy fallback.
    """
    # TODO: Check if user_id exists in faiss_user_to_idx index parameters.
    # Isolate user vector embeddings, call .search() or perform inner products
    # using item_factors_norm.
    # Return lists: (retrieved_item_ids, scores)

    if user_id not in faiss_user_to_idx:
        return [], []

    user_idx = faiss_user_to_idx[user_id]

    if user_idx >= len(user_factors_norm):
        return [], []

    user_vector = user_factors_norm[user_idx:user_idx + 1]

    if FAISS_AVAILABLE and faiss_index is not None:
        scores, item_indices = faiss_index.search(
            user_vector.astype(np.float32),
            top_k
        )

        item_indices = item_indices[0]
        scores = scores[0]

    else:
        similarity_scores = (
            item_factors_norm @ user_vector.T
        ).flatten()

        item_indices = np.argsort(similarity_scores)[::-1][:top_k]
        scores = similarity_scores[item_indices]

    valid_results = [
        (faiss_item_ids[item_index], float(score))
        for item_index, score in zip(item_indices, scores)
        if item_index >= 0 and item_index < len(faiss_item_ids)
    ]

    if not valid_results:
        return [], []

    items, rec_scores = zip(*valid_results)

    return list(items), list(rec_scores)


def get_lfm_recs(user_id, top_k=30):
    """
    Generate predictions for cold-start or low-engagement profiles using
    LightFM Hybrid.
    """
    # TODO: Verify user mapping parameters, build prediction range frameworks,
    # invoke lfm_model.predict() utilizing item_features=lfm_item_features,
    # sort descending, and isolate top_k entries.

    if user_id not in lfm_user_map:
        return [], []

    user_idx = lfm_user_map[user_id]

    item_indices = np.arange(n_lfm_items)

    scores = lfm_model.predict(
        user_ids=np.full(n_lfm_items, user_idx),
        item_ids=item_indices,
        item_features=lfm_item_features,
        num_threads=2
    )

    top_indices = np.argsort(scores)[::-1][:top_k]

    items = [lfm_item_ids_list[index] for index in top_indices]
    top_scores = [float(scores[index]) for index in top_indices]

    return items, top_scores


# ---------------------------------------------------------------------------
# SECTION 4: The Routing Function
# ---------------------------------------------------------------------------
print("\n[4] Defining routing function...")


def route_and_recommend(
    user_id,
    interaction_threshold=3,
    top_k=10,
    lam_mmr=0.5,
    apply_exclusion=True,
    apply_mmr=True
):
    """
    Executes core architectural engine routing rules and controls the
    post-processing pipeline.
    """
    result = {
        'user_id': user_id,
        'engine': None,
        'raw_count': 0,
        'final_recs': [],
        'scores': [],
        'excluded': 0,
        'latency_ms': 0,
    }

    t0 = time.perf_counter()

    # --- STEP 1: ROUTING DECISION ---
    # TODO: Retrieve the number of training interactions (nnz) recorded for this user inside als_user_item.
    # Configure a conditional boolean 'use_als' requiring user registration and nnz >= interaction_threshold.
    interaction_count = 0

    if user_id in als_user_to_idx:
        als_user_idx = als_user_to_idx[user_id]

        if als_user_idx < als_user_item.shape[0]:
            interaction_count = als_user_item[als_user_idx].nnz

    use_als = (
        user_id in als_user_to_idx and
        user_id in faiss_user_to_idx and
        interaction_count >= interaction_threshold
    )

    # --- STEP 2: GET RAW RECOMMENDATIONS ---
    # TODO: Route the request. Call get_als_recs if use_als is true, otherwise branch to get_lfm_recs.
    # Request a candidate multiplier size (e.g., top_k * 3) to allow downstream filtering space.
    candidate_count = top_k * 3

    if use_als:
        result['engine'] = 'ALS + FAISS'
        raw_items, raw_scores = get_als_recs(
            user_id,
            top_k=candidate_count
        )
    else:
        result['engine'] = 'LightFM Hybrid'
        raw_items, raw_scores = get_lfm_recs(
            user_id,
            top_k=candidate_count
        )

    result['raw_count'] = len(raw_items)

    if not raw_items:
        result['latency_ms'] = (time.perf_counter() - t0) * 1000
        return result

    # --- STEP 3: ALREADY-PURCHASED EXCLUSION ---
    if apply_exclusion:
        n_before = len(raw_items)

        raw_items, raw_scores = exclude_purchased(
            raw_items,
            raw_scores,
            user_id
        )

        raw_items = list(raw_items)
        raw_scores = list(raw_scores)

        result['excluded'] = n_before - len(raw_items)

    if not raw_items:
        result['latency_ms'] = (time.perf_counter() - t0) * 1000
        return result

    # --- STEP 4: SCORE NORMALIZATION ---
    norm_scores = min_max_normalize(raw_scores).tolist()

    # --- STEP 5: MMR DIVERSITY RERANKING ---
    if apply_mmr and len(raw_items) > top_k:
        # TODO: Call mmr_rerank passing extracted variables, embeddings, maps, top_k targets, and lambda constraints
        final_items, final_scores = mmr_rerank(
            items=raw_items,
            scores=norm_scores,
            embeddings=item_factors_norm,
            id_to_idx=faiss_id_to_idx,
            top_k=top_k,
            lam=lam_mmr
        )
    else:
        final_items = raw_items[:top_k]
        final_scores = norm_scores[:top_k]

    result['final_recs'] = final_items
    result['scores'] = final_scores
    result['latency_ms'] = (time.perf_counter() - t0) * 1000

    return result


# ---------------------------------------------------------------------------
# SECTION 5: Test the Routing Layer
# ---------------------------------------------------------------------------
print("\n[5] Testing routing layer...")

als_test = [
    user_id
    for user_id in list(als_users)[:20]
    if user_id in faiss_user_to_idx and
    faiss_user_to_idx[user_id] < len(user_factors_norm)
][:3]

lfm_test = [
    user_id
    for user_id in list(lfm_users)[:20]
    if user_id in lfm_user_map
][:3]

# TODO: Execute route_and_recommend calls on sample test arrays (als_test and lfm_test) to print sample performance metrics
for user_id in als_test + lfm_test:
    routing_result = route_and_recommend(
        user_id,
        interaction_threshold=3,
        top_k=10,
        lam_mmr=0.5,
        apply_exclusion=True,
        apply_mmr=True
    )

    print(
        f"\n    User: {user_id}"
        f"\n      Engine       : {routing_result['engine']}"
        f"\n      Raw candidates: {routing_result['raw_count']}"
        f"\n      Excluded     : {routing_result['excluded']}"
        f"\n      Final recs   : {len(routing_result['final_recs'])}"
        f"\n      Latency      : {routing_result['latency_ms']:.2f} ms"
    )

    print(
        f"      Items        : "
        f"{routing_result['final_recs'][:5]}"
    )


# ---------------------------------------------------------------------------
# SECTION 6: MMR Diversity Analysis
# ---------------------------------------------------------------------------
print("\n[6] MMR diversity analysis...")


def intra_list_diversity(items, embeddings, id_to_idx):
    """
    Calculates average distance across an item list:
    D(L) = average(1 - cosine_similarity).
    """
    embs = [
        embeddings[id_to_idx[item_id]]
        for item_id in items
        if item_id in id_to_idx
    ]

    if len(embs) < 2:
        return 0.0

    embs = np.asarray(embs)
    sims = embs @ embs.T

    n_items = len(embs)

    pairs = [
        1 - sims[i, j]
        for i in range(n_items)
        for j in range(i + 1, n_items)
    ]

    return float(np.mean(pairs)) if pairs else 0.0


mmr_diversity_before = 0.0
mmr_diversity_after = 0.0

if als_test:
    test_user = als_test[0]

    # TODO: Run contrast calls isolating a test request with apply_mmr=False vs apply_mmr=True.
    # Pass outputs through intra_list_diversity to measure performance gains.
    recs_without_mmr = route_and_recommend(
        test_user,
        top_k=10,
        apply_exclusion=True,
        apply_mmr=False
    )

    recs_with_mmr = route_and_recommend(
        test_user,
        top_k=10,
        apply_exclusion=True,
        apply_mmr=True
    )

    mmr_diversity_before = intra_list_diversity(
        recs_without_mmr['final_recs'],
        item_factors_norm,
        faiss_id_to_idx
    )

    mmr_diversity_after = intra_list_diversity(
        recs_with_mmr['final_recs'],
        item_factors_norm,
        faiss_id_to_idx
    )

    print(f"    Test user                 : {test_user}")
    print(f"    Diversity without MMR     : {mmr_diversity_before:.4f}")
    print(f"    Diversity with MMR        : {mmr_diversity_after:.4f}")
    print(
        f"    Diversity improvement     : "
        f"{mmr_diversity_after - mmr_diversity_before:+.4f}"
    )


# ---------------------------------------------------------------------------
# SECTION 7: Routing Statistics Across Sample Users
# ---------------------------------------------------------------------------
print("\n[7] Routing statistics across sample users...")

sample_all = (
    [
        user_id
        for user_id in list(als_users)[:50]
        if user_id in faiss_user_to_idx and
        faiss_user_to_idx.get(user_id, 999999) < len(user_factors_norm)
    ] +
    [
        user_id
        for user_id in list(lfm_users)[:50]
        if user_id in lfm_user_map
    ]
)[:100]

# TODO: Iterate over all elements inside sample_all arrays, gather metrics,
# track engine distribution shares, and calculate latency statistics.
routing_results = []

for user_id in sample_all:
    result = route_and_recommend(
        user_id,
        interaction_threshold=3,
        top_k=10,
        lam_mmr=0.5,
        apply_exclusion=True,
        apply_mmr=True
    )

    routing_results.append(result)

engines = [result['engine'] for result in routing_results]
latencies = [result['latency_ms'] for result in routing_results]
exclusions = [result['excluded'] for result in routing_results]

als_count = engines.count('ALS + FAISS')
lfm_count = engines.count('LightFM Hybrid')

valid_latencies = np.asarray(
    [latency for latency in latencies if latency >= 0],
    dtype=float
)

p50_latency = (
    float(np.percentile(valid_latencies, 50))
    if len(valid_latencies) > 0 else 0.0
)

p99_latency = (
    float(np.percentile(valid_latencies, 99))
    if len(valid_latencies) > 0 else 0.0
)

print(f"    Requests evaluated : {len(routing_results):,}")
print(f"    ALS + FAISS routes : {als_count:,}")
print(f"    LightFM routes     : {lfm_count:,}")
print(f"    p50 latency        : {p50_latency:.2f} ms")
print(f"    p99 latency        : {p99_latency:.2f} ms")
print(f"    Avg exclusions     : {np.mean(exclusions):.2f}")


# ---------------------------------------------------------------------------
# SECTION 8: Visualizations
# ---------------------------------------------------------------------------
print("\n[8] Plotting routing analysis...")

os.makedirs("output", exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

fig.suptitle(
    "Lab 3.3: Hybrid Routing Layer Analysis",
    fontsize=12,
    fontweight='bold'
)

# --- Plot 1: Routing Shares Pie Layout ---
# TODO: Plot a pie chart tracing engine traffic breakdowns between ALS and LightFM partitions
# Hint: Use axes[0].pie([als_count, lfm_count], labels=[...], autopct='%1.1f%%')
route_counts = [als_count, lfm_count]

if sum(route_counts) > 0:
    axes[0].pie(
        route_counts,
        labels=['ALS + FAISS', 'LightFM Hybrid'],
        autopct='%1.1f%%',
        colors=['#4C72B0', '#55A868'],
        startangle=90
    )
else:
    axes[0].text(
        0.5,
        0.5,
        'No routing results',
        ha='center',
        va='center'
    )

axes[0].set_title("Routing Engine Distribution")

# --- Plot 2: End-to-End Latency Profiles ---
# TODO: Draw a latency tracking request frequency histogram on axes[1]
# Mark the calculated p99 vertical latency threshold barrier line using axes[1].axvline()
axes[1].hist(
    valid_latencies,
    bins=min(20, max(5, len(valid_latencies))),
    color='#4C72B0',
    edgecolor='white'
)

axes[1].axvline(
    p99_latency,
    color='#D62728',
    linestyle='--',
    linewidth=2,
    label=f'p99 = {p99_latency:.2f} ms'
)

axes[1].set_title("End-to-End Recommendation Latency")
axes[1].set_xlabel("Latency (ms)")
axes[1].set_ylabel("Request Frequency")
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

# --- Plot 3: Exclusion Volume Trackers ---
# TODO: Render a frequency bar histogram showing counts of items removed per request due to prior purchases
# Hint: Use axes[2].hist(exclusions, bins=...)
axes[2].hist(
    exclusions,
    bins=np.arange(
        min(exclusions, default=0),
        max(exclusions, default=0) + 2
    ) - 0.5,
    color='#DD8452',
    edgecolor='white'
)

axes[2].set_title("Purchased Item Exclusions")
axes[2].set_xlabel("Items Removed per Request")
axes[2].set_ylabel("Request Frequency")
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("output/03_routing_analysis.png", dpi=150, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------------------
# SECTION 9: Save Routing Artifacts
# ---------------------------------------------------------------------------
print("\n[9] Saving routing artifacts...")

# TODO: Export calculated summaries, latencies vectors, lists, and metrics down to a serialization storage target
# Target Path: "data/routing_artifacts.pkl"
routing_artifacts = {
    'routing_results': routing_results,
    'sample_users': sample_all,
    'als_count': als_count,
    'lightfm_count': lfm_count,
    'latencies_ms': latencies,
    'exclusions': exclusions,
    'p50_latency_ms': p50_latency,
    'p99_latency_ms': p99_latency,
    'mmr_diversity_before': mmr_diversity_before,
    'mmr_diversity_after': mmr_diversity_after,
    'mmr_diversity_improvement': (
        mmr_diversity_after - mmr_diversity_before
    ),
    'faiss_available': FAISS_AVAILABLE,
    'routing_config': {
        'interaction_threshold': 3,
        'top_k': 10,
        'mmr_lambda': 0.5,
        'exclusion_enabled': True,
        'mmr_enabled': True
    }
}

artifact_path = "data/routing_artifacts.pkl"
temp_artifact_path = "data/routing_artifacts_temp.pkl"

with open(temp_artifact_path, "wb") as file:
    pickle.dump(
        routing_artifacts,
        file,
        protocol=pickle.HIGHEST_PROTOCOL
    )

os.replace(temp_artifact_path, artifact_path)

print("    Saved -> data/routing_artifacts.pkl")
print("    Move to: 04_ab_protocol.py")
