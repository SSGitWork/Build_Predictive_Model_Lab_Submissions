# =============================================================================
# MODULE 4 | LAB 4.2
# File: 02_monitoring.py
# Purpose: Build a recommendation engine quality assurance monitor tracking
#          Catalog Coverage, Self-Information Novelty, and Intra-list Diversity.
#          Simulate cold-start catalog updates and degradation alerting logic.
# Saras AI Institute | Build Predictive Models & Modern Recommenders
# =============================================================================

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("  MODULE 4 | LAB 4.2")
print("  Recommendation Quality Monitoring")
print("  Coverage · Novelty · Catalog Change Simulation")
print("=" * 60)


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TOP_K = 10
N_SAMPLE_USERS = 100
N_NEW_ITEMS = 500
ALERT_TOLERANCE = 0.10

rng = np.random.default_rng(RANDOM_STATE)


# ---------------------------------------------------------------------------
# Helper Function: Safe Artifact Loading
# ---------------------------------------------------------------------------
def load_pickle_checked(path):
    """Load a non-empty pickle artifact with clear error messages."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required artifact: {path}. "
            "Run the prerequisite lab first."
        )

    if os.path.getsize(path) == 0:
        raise ValueError(
            f"Artifact is empty: {path}. "
            "Rerun the prerequisite lab."
        )

    try:
        with open(path, "rb") as file:
            return pickle.load(file)

    except EOFError as error:
        raise ValueError(
            f"Artifact is incomplete or corrupted: {path}."
        ) from error


# ---------------------------------------------------------------------------
# SECTION 1: Load Artifacts
# ---------------------------------------------------------------------------
print("\n[1] Loading artifacts...")

als_art = load_pickle_checked("data/als_artifacts.pkl")
lfm_art = load_pickle_checked("data/lightfm_serving.pkl")
fai_art = load_pickle_checked("data/faiss_artifacts.pkl")

als_model = als_art['model']
als_user_to_idx = als_art['user_to_idx']
als_item_ids = np.asarray(als_art['item_ids'])

# Prefer the temporal training matrix when it is available.
als_user_item = als_art.get(
    'train_user_item_matrix',
    als_art['user_item_matrix']
)

item_factors_norm = np.asarray(fai_art['item_factors_norm'])
user_factors_norm = np.asarray(fai_art['user_factors_norm'])
faiss_item_ids = np.asarray(fai_art['item_ids'])
faiss_user_to_idx = fai_art['user_to_idx']

# Serving artifact avoids loading the full multi-GB LightFM training artifact.
lfm_model = lfm_art['model_hybrid']
lfm_item_f = lfm_art['item_features_matrix']
lfm_item_ids_list = lfm_art['lfm_item_ids_list']
n_lfm_items = lfm_art['n_lfm_items']

events = pd.read_csv("data/events.csv")
events['datetime'] = pd.to_datetime(events['timestamp'], unit='ms')

print(f"    ALS items              : {len(als_item_ids):,}")
print(f"    LightFM serving items  : {n_lfm_items:,}")
print(f"    ALS embedding shape    : {item_factors_norm.shape}")


# ---------------------------------------------------------------------------
# SECTION 2: Define Monitoring Metrics
# ---------------------------------------------------------------------------
print("\n[2] Defining monitoring metrics...")


def compute_coverage(all_recs, catalog_size):
    """
    Calculates the fraction of total catalog items surface-recommending to users.
    Formula: Coverage = | union(R_u) | / |C|
    """
    # TODO: Isolate unique item recommendations by parsing the list-of-lists parameter (all_recs) into a set boundary.
    # Return unique items count divided by catalog_size.
    if catalog_size <= 0:
        return 0.0

    unique_recommended_items = set()

    for recommendation_list in all_recs:
        unique_recommended_items.update(recommendation_list)

    return len(unique_recommended_items) / catalog_size


def compute_novelty(all_recs, item_popularity):
    """
    Calculates macro informational unexpectedness scores based on self-information scale.
    Formula: Novelty = average(-log2(P(i)))
    """
    # TODO: Iterate over every item across recommendations inside all_recs.
    # Extract structural probability P(i) from item_popularity, falling back to 1e-10 if unrecorded.
    # Increment tracking metrics with self-information logs: -np.log2(pop). Return the mean.
    novelty_scores = []

    for recommendation_list in all_recs:
        for item_id in recommendation_list:
            probability = item_popularity.get(item_id, 1e-10)
            probability = max(probability, 1e-10)

            novelty_scores.append(-np.log2(probability))

    return float(np.mean(novelty_scores)) if novelty_scores else 0.0


def compute_intra_list_diversity(all_recs, item_factors, item_id_map):
    """
    Calculates macro pairwise distance parameters tracking internal similarity alignments.
    Formula:
    D(R_u) = average(1 - cosine_similarity(i, j)) for all pairs i != j.
    """
    # TODO: Loop over each individual recommendation list inside all_recs container arrays.
    # Extract normalized embedding matrix layouts mapping items using item_id_map and item_factors.
    # Compute pair distance weights using matrix transposition inner dot products (1 - sim).
    # Return the global across-list mean diversity score.
    diversity_scores = []

    for recommendation_list in all_recs:
        valid_item_indices = [
            item_id_map[item_id]
            for item_id in recommendation_list
            if item_id in item_id_map
        ]

        if len(valid_item_indices) < 2:
            continue

        embedding_matrix = item_factors[valid_item_indices]

        similarity_matrix = embedding_matrix @ embedding_matrix.T

        upper_triangle_indices = np.triu_indices(
            len(valid_item_indices),
            k=1
        )

        pairwise_similarities = similarity_matrix[upper_triangle_indices]

        if len(pairwise_similarities) > 0:
            diversity_scores.append(
                float(np.mean(1 - pairwise_similarities))
            )

    return float(np.mean(diversity_scores)) if diversity_scores else 0.0


print("    Metrics processing structure : initialized")


# ---------------------------------------------------------------------------
# SECTION 3: Compute Item Popularity
# ---------------------------------------------------------------------------
print("\n[3] Computing item popularity...")

# TODO: Compute baseline transaction count ratios to discover structural background probability distributions
# Hint: Calculate size counts grouped by 'itemid' across the events data frame and divide by total events size
item_counts = events.groupby('itemid').size()
total_events = len(events)

item_popularity = (
    item_counts / total_events
).to_dict()

# Map items indices back to numerical factor locations
item_id_to_emb = {
    item_id: index
    for index, item_id in enumerate(faiss_item_ids)
}

print(f"    Items with historical popularity: {len(item_popularity):,}")


# ---------------------------------------------------------------------------
# SECTION 4: Generate Baseline Recommendations
# ---------------------------------------------------------------------------
print("\n[4] Generating baseline recommendations (100 users)...")


def get_als_recs_simple(user_id, top_k=10):
    """Generates standard lookups targeting the base user factor spaces."""
    if user_id not in faiss_user_to_idx:
        return []

    user_index = faiss_user_to_idx[user_id]

    if user_index >= len(user_factors_norm):
        return []

    user_vector = user_factors_norm[user_index:user_index + 1].astype(
        np.float32
    )

    raw_scores = item_factors_norm @ user_vector.T

    top_k_indices = np.argsort(
        raw_scores.flatten()
    )[::-1][:top_k]

    return [
        int(faiss_item_ids[index])
        for index in top_k_indices
        if 0 <= index < len(faiss_item_ids)
    ]


available_users = [
    user_id
    for user_id in faiss_user_to_idx.keys()
    if faiss_user_to_idx[user_id] < len(user_factors_norm)
]

sample_users = available_users[:N_SAMPLE_USERS]

# TODO: Compile baseline recommendation tracking blocks by mapping your get_als_recs_simple loop across sample_users
baseline_recs = [
    get_als_recs_simple(user_id, top_k=TOP_K)
    for user_id in sample_users
]

baseline_recs = [
    recommendations
    for recommendations in baseline_recs
    if recommendations
]

catalog_size = len(faiss_item_ids)

# TODO: Pass baseline arrays down through compute_coverage, compute_novelty, and compute_intra_list_diversity methods
baseline_coverage = compute_coverage(
    baseline_recs,
    catalog_size
)

baseline_novelty = compute_novelty(
    baseline_recs,
    item_popularity
)

baseline_diversity = compute_intra_list_diversity(
    baseline_recs,
    item_factors_norm,
    item_id_to_emb
)

print(f"    Evaluation users      : {len(baseline_recs):,}")
print(f"    Baseline Coverage     : {baseline_coverage:.4f}")
print(f"    Baseline Novelty      : {baseline_novelty:.4f}")
print(f"    Baseline Diversity    : {baseline_diversity:.4f}")


# ---------------------------------------------------------------------------
# SECTION 5: Simulate Catalog Change — Add 500 New Items
# ---------------------------------------------------------------------------
print("\n[5] Simulating catalog change — adding 500 new items...")


def generate_unit_embeddings(n_items, embedding_dimension, generator):
    """Generate random L2-normalized vectors to represent unseen new items."""
    embeddings = generator.normal(
        loc=0.0,
        scale=1.0,
        size=(n_items, embedding_dimension)
    ).astype(np.float32)

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    norms[norms == 0] = 1.0

    return embeddings / norms


def get_recs_from_embeddings(
    user_id,
    item_embeddings,
    item_catalog,
    top_k=10
):
    """Generate top-K recommendations using an extended item-embedding matrix."""
    if user_id not in faiss_user_to_idx:
        return []

    user_index = faiss_user_to_idx[user_id]

    if user_index >= len(user_factors_norm):
        return []

    user_vector = user_factors_norm[user_index:user_index + 1]

    scores = item_embeddings @ user_vector.T

    top_indices = np.argsort(
        scores.flatten()
    )[::-1][:top_k]

    return [
        int(item_catalog[index])
        for index in top_indices
        if 0 <= index < len(item_catalog)
    ]


# TODO: Model cold-start catalog updates. Append 500 un-interacted item IDs onto the existing catalog structure.
# Generate mock unit-norm random embedding vector configurations matching your baseline embedding layout sizes.
max_existing_item_id = int(np.max(faiss_item_ids))

new_item_ids = np.arange(
    max_existing_item_id + 1,
    max_existing_item_id + N_NEW_ITEMS + 1,
    dtype=np.int64
)

new_item_embeddings = generate_unit_embeddings(
    n_items=N_NEW_ITEMS,
    embedding_dimension=item_factors_norm.shape[1],
    generator=rng
)

extended_item_ids = np.concatenate([
    faiss_item_ids,
    new_item_ids
])

extended_item_embs = np.vstack([
    item_factors_norm,
    new_item_embeddings
]).astype(np.float32)

extended_id_to_emb = {
    item_id: index
    for index, item_id in enumerate(extended_item_ids)
}

# TODO: Generate recommendations using your extended matrices boundaries over sample_users
extended_recs = [
    get_recs_from_embeddings(
        user_id,
        item_embeddings=extended_item_embs,
        item_catalog=extended_item_ids,
        top_k=TOP_K
    )
    for user_id in sample_users
]

extended_recs = [
    recommendations
    for recommendations in extended_recs
    if recommendations
]

extended_catalog_size = len(extended_item_ids)

new_item_popularity = {
    item_id: 1e-10
    for item_id in new_item_ids
}

extended_popularity = {
    **item_popularity,
    **new_item_popularity
}

# TODO: Evaluate performance metrics over the extended recommendation logs
extended_coverage = compute_coverage(
    extended_recs,
    extended_catalog_size
)

extended_novelty = compute_novelty(
    extended_recs,
    extended_popularity
)

extended_diversity = compute_intra_list_diversity(
    extended_recs,
    extended_item_embs,
    extended_id_to_emb
)

print(f"\n    Extended Coverage     : {extended_coverage:.4f}")
print(f"    Extended Novelty      : {extended_novelty:.4f}")
print(f"    Extended Diversity    : {extended_diversity:.4f}")


# ---------------------------------------------------------------------------
# SECTION 6: Degradation Report & Alerting
# ---------------------------------------------------------------------------
print("\n[6] Quality degradation report:")
print(f"\n    {'Metric':<25} {'Baseline':>10} {'After Change':>12} {'Delta':>10} {'Status':>15}")
print(f"    {'-' * 78}")

metric_rows = [
    ('Coverage', baseline_coverage, extended_coverage),
    ('Novelty', baseline_novelty, extended_novelty),
    ('Diversity', baseline_diversity, extended_diversity)
]

monitoring_alerts = []

# TODO: Check performance variance boundaries. Loop across calculation arrays comparing changes.
# Trigger a warning flag or validation string print update if metric drops cross a 10% tolerance boundary.
for metric_name, baseline_value, changed_value in metric_rows:
    delta = changed_value - baseline_value

    relative_change = (
        delta / baseline_value
        if baseline_value != 0
        else 0.0
    )

    if baseline_value > 0 and relative_change < -ALERT_TOLERANCE:
        status = 'ALERT: DROP >10%'
        monitoring_alerts.append(metric_name)
    else:
        status = 'OK'

    print(
        f"    {metric_name:<25} "
        f"{baseline_value:>10.4f} "
        f"{changed_value:>12.4f} "
        f"{delta:>+10.4f} "
        f"{status:>15}"
    )

if monitoring_alerts:
    print(
        f"\n    ALERT TRIGGERED: Quality degradation detected in "
        f"{', '.join(monitoring_alerts)}."
    )
else:
    print("\n    All monitored quality metrics remain within tolerance.")


# ---------------------------------------------------------------------------
# SECTION 7: New Item Coverage Analysis
# ---------------------------------------------------------------------------
print("\n[7] New item coverage analysis...")

# TODO: Intersect extended recommended tracking vectors against new_item_ids to trace conversion ratios
# Calculate what percentage of cold items were reached by the matrix lookup engine
recommended_extended_items = set()

for recommendations in extended_recs:
    recommended_extended_items.update(recommendations)

reached_new_items = recommended_extended_items.intersection(
    set(new_item_ids)
)

new_item_coverage = (
    len(reached_new_items) / len(new_item_ids)
    if len(new_item_ids) > 0
    else 0.0
)

print(f"    New items added        : {len(new_item_ids):,}")
print(f"    New items recommended  : {len(reached_new_items):,}")
print(f"    New item coverage      : {new_item_coverage:.2%}")


# ---------------------------------------------------------------------------
# SECTION 8: Monitoring Over Time Simulation
# ---------------------------------------------------------------------------
print("\n[8] Simulating monitoring over 10 time windows...")

time_windows = range(1, 11)
coverage_track = []
novelty_track = []
diversity_track = []
cold_item_coverage_track = []

# TODO: Construct a loop over window intervals gradually growing directory catalogs (e.g., n_new = window * 50).
# Record sequential metric shifts inside tracking lists to simulate production time series monitors.
for window in time_windows:
    n_new = window * 50

    window_item_ids = new_item_ids[:n_new]
    window_item_embs = new_item_embeddings[:n_new]

    window_catalog = np.concatenate([
        faiss_item_ids,
        window_item_ids
    ])

    window_embeddings = np.vstack([
        item_factors_norm,
        window_item_embs
    ]).astype(np.float32)

    window_item_id_map = {
        item_id: index
        for index, item_id in enumerate(window_catalog)
    }

    window_recs = [
        get_recs_from_embeddings(
            user_id,
            item_embeddings=window_embeddings,
            item_catalog=window_catalog,
            top_k=TOP_K
        )
        for user_id in sample_users
    ]

    window_recs = [
        recommendations
        for recommendations in window_recs
        if recommendations
    ]

    window_popularity = {
        **item_popularity,
        **{
            item_id: 1e-10
            for item_id in window_item_ids
        }
    }

    window_coverage = compute_coverage(
        window_recs,
        len(window_catalog)
    )

    window_novelty = compute_novelty(
        window_recs,
        window_popularity
    )

    window_diversity = compute_intra_list_diversity(
        window_recs,
        window_embeddings,
        window_item_id_map
    )

    window_recommended_items = set()

    for recommendations in window_recs:
        window_recommended_items.update(recommendations)

    reached_window_new_items = window_recommended_items.intersection(
        set(window_item_ids)
    )

    window_cold_item_coverage = (
        len(reached_window_new_items) / len(window_item_ids)
        if len(window_item_ids) > 0
        else 0.0
    )

    coverage_track.append(window_coverage)
    novelty_track.append(window_novelty)
    diversity_track.append(window_diversity)
    cold_item_coverage_track.append(window_cold_item_coverage)

    print(
        f"    Window {window:>2}: "
        f"catalog +{n_new:>3} | "
        f"coverage={window_coverage:.4f} | "
        f"novelty={window_novelty:.4f}"
    )


# ---------------------------------------------------------------------------
# SECTION 9: Visualizations
# ---------------------------------------------------------------------------
print("\n[9] Plotting monitoring dashboard...")

os.makedirs("output", exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

fig.suptitle(
    "Lab 4.2: Recommendation Quality Monitoring\n"
    "Coverage · Novelty · Catalog Change Impact",
    fontsize=12,
    fontweight='bold'
)

# --- Plot 1: Before vs After Metrics Shifts Comparison Chart ---
# TODO: Draw an adjacent bar layout on axes[0,0] detailing performance changes before vs after cold-start updates
metric_labels = ['Coverage', 'Novelty', 'Diversity']

baseline_values = [
    baseline_coverage,
    baseline_novelty,
    baseline_diversity
]

extended_values = [
    extended_coverage,
    extended_novelty,
    extended_diversity
]

x = np.arange(len(metric_labels))
bar_width = 0.35

axes[0, 0].bar(
    x - bar_width / 2,
    baseline_values,
    width=bar_width,
    label='Baseline',
    color='#4C72B0'
)

axes[0, 0].bar(
    x + bar_width / 2,
    extended_values,
    width=bar_width,
    label='After +500 New Items',
    color='#DD8452'
)

axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(metric_labels)
axes[0, 0].set_title("Quality Metrics: Before vs After")
axes[0, 0].set_ylabel("Metric Score")
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# --- Plot 2: Simulated Coverage Scaling Trajectory ---
# TODO: Draw a line trace map plotting coverage changes over time intervals on axes[0,1]
# Draw an SLA alert limit indicator marking a 10% baseline performance drop via axes[0,1].axhline()
coverage_alert_threshold = baseline_coverage * (1 - ALERT_TOLERANCE)

axes[0, 1].plot(
    list(time_windows),
    coverage_track,
    marker='o',
    linewidth=2,
    color='#4C72B0',
    label='Catalog Coverage'
)

axes[0, 1].axhline(
    coverage_alert_threshold,
    color='#D62728',
    linestyle='--',
    linewidth=2,
    label='10% Drop Alert Threshold'
)

axes[0, 1].set_title("Coverage During Catalog Growth")
axes[0, 1].set_xlabel("Monitoring Time Window")
axes[0, 1].set_ylabel("Catalog Coverage")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# --- Plot 3: Novelty Metric Scale Trajectory ---
# TODO: Map shifts across calculated unexpectedness score metrics over window steps on axes[1,0]
axes[1, 0].plot(
    list(time_windows),
    novelty_track,
    marker='s',
    linewidth=2,
    color='#55A868',
    label='Novelty'
)

axes[1, 0].axhline(
    baseline_novelty,
    color='#4C72B0',
    linestyle='--',
    linewidth=2,
    label='Baseline Novelty'
)

axes[1, 0].set_title("Novelty During Catalog Growth")
axes[1, 0].set_xlabel("Monitoring Time Window")
axes[1, 0].set_ylabel("Self-Information Novelty")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# --- Plot 4: Cold Item Conversion Segment Shares ---
# TODO: Compile a pie chart tracking segment proportions detailing reached vs unreached cold candidate blocks on axes[1,1]
reached_count = len(reached_new_items)
unreached_count = len(new_item_ids) - reached_count

if len(new_item_ids) > 0:
    axes[1, 1].pie(
        [reached_count, unreached_count],
        labels=['New Items Reached', 'New Items Not Reached'],
        autopct='%1.1f%%',
        colors=['#55A868', '#D3D3D3'],
        startangle=90
    )

axes[1, 1].set_title("Cold-Start New Item Exposure")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(
    "output/02_monitoring_dashboard.png",
    dpi=150,
    bbox_inches='tight'
)
plt.show()

print("    Saved -> output/02_monitoring_dashboard.png")


# ---------------------------------------------------------------------------
# SECTION 10: Save Monitoring Metrics
# ---------------------------------------------------------------------------
print("\n[10] Saving monitoring artifacts...")

monitoring_artifacts = {
    'baseline_metrics': {
        'coverage': baseline_coverage,
        'novelty': baseline_novelty,
        'diversity': baseline_diversity
    },
    'extended_metrics': {
        'coverage': extended_coverage,
        'novelty': extended_novelty,
        'diversity': extended_diversity
    },
    'new_item_metrics': {
        'new_items_added': len(new_item_ids),
        'new_items_reached': len(reached_new_items),
        'new_item_coverage': new_item_coverage
    },
    'monitoring_alerts': monitoring_alerts,
    'time_windows': list(time_windows),
    'coverage_track': coverage_track,
    'novelty_track': novelty_track,
    'diversity_track': diversity_track,
    'cold_item_coverage_track': cold_item_coverage_track,
    'config': {
        'top_k': TOP_K,
        'sample_users': len(sample_users),
        'new_items': N_NEW_ITEMS,
        'alert_tolerance': ALERT_TOLERANCE,
        'random_state': RANDOM_STATE
    }
}

with open("data/monitoring_artifacts.pkl", "wb") as file:
    pickle.dump(
        monitoring_artifacts,
        file,
        protocol=pickle.HIGHEST_PROTOCOL
    )

print("    Saved -> data/monitoring_artifacts.pkl")

print("\n" + "=" * 60)
print("  LAB 4.2 COMPLETE — MONITORING DASHBOARD")
print("=" * 60)
