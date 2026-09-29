# =============================================================================
# MODULE 3 | LAB 3.2
# File: 02_faiss_index.py
# Purpose: Build and benchmark a FAISS vector index using ALS embeddings
#          to achieve low-latency k-NN retrieval spans.
# Saras AI Institute | Build Predictive Models & Modern Recommenders
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import warnings

warnings.filterwarnings('ignore')

try:
    import faiss
    FAISS_AVAILABLE = True
    print("FAISS imported successfully")
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not installed. Run: pip install faiss-cpu")
    print("Falling back to numpy brute-force search.")

print("=" * 60)
print("  MODULE 3 | LAB 3.2")
print("  FAISS Index for ALS Embeddings")
print("  Target: p99 retrieval latency < 20ms")
print("=" * 60)


# ---------------------------------------------------------------------------
# SECTION 1: Load ALS Artifacts from Lab 3.1
# ---------------------------------------------------------------------------
print("\n[1] Loading ALS artifacts from Lab 3.1...")

als_artifact_path = "data/als_artifacts.pkl"

if not os.path.exists(als_artifact_path):
    raise FileNotFoundError(
        "data/als_artifacts.pkl was not found. "
        "Run Lab 3.1 before running this script."
    )

if os.path.getsize(als_artifact_path) == 0:
    raise ValueError(
        "data/als_artifacts.pkl is empty. "
        "Rerun Lab 3.1 completely."
    )

with open(als_artifact_path, "rb") as f:
    als = pickle.load(f)

# Extract parameters from the loaded dictionary package
item_factors = als['item_factors']
user_factors = als['user_factors']
item_ids = np.asarray(als['item_ids'])
user_ids = np.asarray(als['user_ids'])
user_to_idx = als['user_to_idx']
item_to_idx = als['item_to_idx']

# Use the training matrix when available. This is retained for Lab 3.3.
train_user_item = als.get(
    'train_user_item_matrix',
    als.get('user_item_matrix')
)

print(f"    Item factors shape : {item_factors.shape}")
print(f"    User factors shape : {user_factors.shape}")
print(f"    Embedding dim      : {item_factors.shape[1]}")


# ---------------------------------------------------------------------------
# SECTION 2: Normalize Embeddings
# ---------------------------------------------------------------------------
print("\n[2] Normalizing embeddings...")

def normalize_vectors(vectors):
    """
    Normalize spatial embedding arrays to unit length (L2 norm = 1).
    This ensures that the inner product matches cosine similarity measurements.
    """
    # TODO: Calculate the L2 norm for each row vector across the horizontal axis
    # Hint: Use np.linalg.norm(..., axis=1, keepdims=True)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    # TODO: Replace any calculated zero norm indices with 1 to avoid divide-by-zero errors
    norms[norms == 0] = 1.0

    # TODO: Divide the raw vectors by their calculated norms and cast the matrix to np.float32
    return (vectors / norms).astype(np.float32)

# TODO: Normalize item and user factor matrices using your custom normalize_vectors function
item_factors_norm = normalize_vectors(item_factors)
user_factors_norm = normalize_vectors(user_factors)

DIM = item_factors_norm.shape[1]
N_ITEMS = len(item_ids)

print(f"    Normalized items   : {item_factors_norm.shape}")
print(f"    Normalized users   : {user_factors_norm.shape}")


# ---------------------------------------------------------------------------
# SECTION 3: Build FAISS Indices
# ---------------------------------------------------------------------------
print("\n[3] Building FAISS indices...")

if FAISS_AVAILABLE and N_ITEMS > 0:
    # --- Index 1: Exact Flat Inner Product Search ---
    print("    Building IndexFlatIP (exact search)...")
    t0 = time.time()

    # TODO: Instantiate a flat inner product FAISS index using vector dimension limits
    # Hint: Call faiss.IndexFlatIP(DIM)
    index_exact = faiss.IndexFlatIP(DIM)

    # TODO: Add the normalized item factors matrix to the exact index object
    index_exact.add(item_factors_norm)

    exact_build_time = time.time() - t0
    print(f"    IndexFlatIP built in   : {exact_build_time:.3f}s")

    # --- Index 2: Approximate Inverted File Index (IVF) ---
    print("\n    Building IndexIVFFlat (approximate search)...")

    # Ensure nlist is at least 1 and never exceeds item count.
    nlist = max(1, min(50, N_ITEMS // 10))
    nprobe = min(10, nlist)

    t0 = time.time()

    # TODO: Create an internal IndexFlatIP quantizer instance to act as the cluster centroid assessor
    quantizer = faiss.IndexFlatIP(DIM)

    # TODO: Instantiate an inverted file structural cell (IndexIVFFlat) linking your quantizer map
    # Arguments: quantizer, DIM, nlist, faiss.METRIC_INNER_PRODUCT
    index_approx = faiss.IndexIVFFlat(
        quantizer,
        DIM,
        nlist,
        faiss.METRIC_INNER_PRODUCT
    )

    # TODO: Train the approximate index using the structural item_factors_norm distribution
    index_approx.train(item_factors_norm)

    # TODO: Populate the trained approximate index cells by calling .add() with item_factors_norm
    index_approx.add(item_factors_norm)

    # TODO: Configure the index lookup parameter by assigning index_approx.nprobe to nprobe
    index_approx.nprobe = nprobe

    approx_build_time = time.time() - t0
    print(f"    IndexIVFFlat built in  : {approx_build_time:.3f}s")
    print(f"    IVF clusters (nlist)   : {nlist}")
    print(f"    IVF probes (nprobe)    : {nprobe}")

else:
    index_exact = None
    index_approx = None
    exact_build_time = 0.0
    approx_build_time = 0.0
    nlist = 0
    nprobe = 0
    print("    Using numpy brute-force (FAISS not available or skipped)")


# ---------------------------------------------------------------------------
# SECTION 4: Latency Benchmarking
# ---------------------------------------------------------------------------
print("\n[4] Benchmarking retrieval latency...")

N_QUERIES = min(1000, len(user_factors_norm))
K_RESULTS = 10

np.random.seed(42)

if N_QUERIES > 0:
    sample_idx = np.random.choice(
        len(user_factors_norm),
        N_QUERIES,
        replace=False
    )
    sample_vectors = user_factors_norm[sample_idx]
else:
    sample_vectors = np.empty((0, DIM), dtype=np.float32)

def benchmark_search(search_fn, query_vectors, n_queries, label):
    """
    Executes isolated search lookups sequentially to extract percentile performance spans.
    """
    latencies = []

    for i in range(n_queries):
        # Isolate a single query vector slice of shape (1, DIM)
        query = query_vectors[i:i + 1]

        t0 = time.perf_counter()
        search_fn(query)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)

    if len(latencies) == 0:
        print(f"\n    {label}: no query vectors available")
        return latencies

    p99 = np.percentile(latencies, 99)

    print(f"\n    {label}:")
    print(f"      p50 : {np.percentile(latencies, 50):.3f} ms")
    print(f"      p99 : {p99:.3f} ms")

    return latencies

if FAISS_AVAILABLE and index_exact is not None and index_approx is not None:
    # TODO: Pass index_exact.search lookup lambda expressions into the benchmark engine
    # Syntax hint for lambda: lambda q: index_exact.search(q, K_RESULTS)
    exact_latencies = benchmark_search(
        lambda q: index_exact.search(q, K_RESULTS),
        sample_vectors,
        N_QUERIES,
        "FAISS IndexFlatIP (Exact)"
    )

    # TODO: Pass index_approx.search lookup lambda expressions into the benchmark engine
    approx_latencies = benchmark_search(
        lambda q: index_approx.search(q, K_RESULTS),
        sample_vectors,
        N_QUERIES,
        "FAISS IndexIVFFlat (Approximate)"
    )

else:
    # NumPy fallback routine if FAISS dependencies are not installed
    def numpy_search(query):
        scores = item_factors_norm @ query.T
        top_k = np.argsort(scores.flatten())[::-1][:K_RESULTS]
        return top_k, scores[top_k]

    exact_latencies = benchmark_search(
        numpy_search,
        sample_vectors,
        N_QUERIES,
        "NumPy Brute-Force"
    )

    approx_latencies = exact_latencies.copy()


# ---------------------------------------------------------------------------
# SECTION 5: Accuracy — Exact vs Approximate
# ---------------------------------------------------------------------------
print("\n[5] ANN accuracy vs exact search (100 queries)...")

if FAISS_AVAILABLE and index_exact is not None and index_approx is not None:
    recall_scores = []

    for i in range(min(100, N_QUERIES)):
        q = sample_vectors[i:i + 1]

        # TODO: Run identical query items across both your index_exact and index_approx indices
        _, exact_ids = index_exact.search(q, K_RESULTS)
        _, approx_ids = index_approx.search(q, K_RESULTS)

        # TODO: Intersect item index sets to find the count of overlapping recommended items
        # Calculate recall score by dividing overlapping matches by K_RESULTS
        overlap = len(set(exact_ids[0]) & set(approx_ids[0]))
        recall_scores.append(overlap / K_RESULTS)

    mean_recall = float(np.mean(recall_scores)) if recall_scores else 0.0
    print(f"    Mean Recall@10 (ANN vs Exact): {mean_recall:.4f}")

else:
    mean_recall = 1.0
    print("    Skipped — FAISS not available")


# ---------------------------------------------------------------------------
# SECTION 6: Sample Recommendations via FAISS
# ---------------------------------------------------------------------------
print("\n[6] Sample recommendations via FAISS index...")

if user_ids is not None and user_factors_norm is not None:
    sample_users = [
        user_ids[i]
        for i in range(min(3, len(user_ids)))
    ]

    for user_id in sample_users:
        user_idx = user_to_idx[user_id]
        user_vec = user_factors_norm[user_idx:user_idx + 1]

        # TODO: Query your vector index using the user_vec embedding to retrieve the closest matching items
        # Map indices back to original item IDs from the dataset catalog and print rankings
        if FAISS_AVAILABLE and index_approx is not None:
            scores, item_indices = index_approx.search(user_vec, K_RESULTS)
            item_indices = item_indices[0]
            scores = scores[0]

        elif FAISS_AVAILABLE and index_exact is not None:
            scores, item_indices = index_exact.search(user_vec, K_RESULTS)
            item_indices = item_indices[0]
            scores = scores[0]

        else:
            scores = (item_factors_norm @ user_vec.T).flatten()
            item_indices = np.argsort(scores)[::-1][:K_RESULTS]
            scores = scores[item_indices]

        valid_results = [
            (item_index, score)
            for item_index, score in zip(item_indices, scores)
            if item_index >= 0
        ]

        print(f"\n    User {user_id} recommendations:")

        for rank, (item_index, score) in enumerate(valid_results[:5], start=1):
            print(
                f"      {rank}. Item {item_ids[item_index]} "
                f"| cosine score = {score:.4f}"
            )


# ---------------------------------------------------------------------------
# SECTION 7: Visualizations
# ---------------------------------------------------------------------------
print("\n[7] Plotting latency analysis...")

os.makedirs("output", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fig.suptitle(
    "Lab 3.2: FAISS Index — Retrieval Latency Benchmarking",
    fontsize=12,
    fontweight='bold'
)

# --- Plot 1: Latency Distributions ---
# TODO: Plot histograms on axes[0] mapping exact_latencies against approx_latencies distributions
# Highlight a vertical baseline indicator line representing your 20ms target rule using axes[0].axvline()
axes[0].hist(
    exact_latencies,
    bins=40,
    alpha=0.65,
    color='#4C72B0',
    label='Exact Search'
)

axes[0].hist(
    approx_latencies,
    bins=40,
    alpha=0.65,
    color='#55A868',
    label='Approximate Search'
)

axes[0].axvline(
    20,
    color='#D62728',
    linestyle='--',
    linewidth=2,
    label='20ms Target'
)

axes[0].set_title("Latency Distribution")
axes[0].set_xlabel("Query Latency (ms)")
axes[0].set_ylabel("Frequency")
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# --- Plot 2: Percentile Metrics Comparison ---
# TODO: Group exact vs approximate latency values across matching percentile bins (p50, p75, p90, p95, p99)
# Render these summaries on axes[1] as a comparative bar layout chart
percentiles = [50, 75, 90, 95, 99]
percentile_labels = ['p50', 'p75', 'p90', 'p95', 'p99']

exact_percentiles = np.percentile(exact_latencies, percentiles)
approx_percentiles = np.percentile(approx_latencies, percentiles)

x = np.arange(len(percentiles))
width = 0.36

axes[1].bar(
    x - width / 2,
    exact_percentiles,
    width,
    color='#4C72B0',
    label='Exact'
)

axes[1].bar(
    x + width / 2,
    approx_percentiles,
    width,
    color='#55A868',
    label='Approximate'
)

axes[1].axhline(
    20,
    color='#D62728',
    linestyle='--',
    linewidth=2,
    label='20ms Target'
)

axes[1].set_xticks(x)
axes[1].set_xticklabels(percentile_labels)
axes[1].set_title("Latency Percentiles")
axes[1].set_xlabel("Percentile")
axes[1].set_ylabel("Latency (ms)")
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("output/02_faiss_latency.png", dpi=150, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------------------
# SECTION 8: Save FAISS Artifacts for Lab 3.3
# ---------------------------------------------------------------------------
print("\n[8] Saving FAISS artifacts...")

faiss_index_path = "data/faiss_index.bin"

if FAISS_AVAILABLE and index_approx is not None:
    # TODO: Write your configured index binary structure to disk space
    # Hint: Call faiss.write_index(index_approx, "data/faiss_index.bin")
    faiss.write_index(index_approx, faiss_index_path)
    print(f"    Saved -> {faiss_index_path}")

# TODO: Save tracking parameters, normalization matrices, validation scores, and flags to an output pickle index package
# Target Path: "data/faiss_artifacts.pkl"
faiss_artifacts = {
    'faiss_available': FAISS_AVAILABLE,
    'faiss_index_path': faiss_index_path if FAISS_AVAILABLE else None,
    'item_ids': item_ids,
    'user_ids': user_ids,
    'user_to_idx': user_to_idx,
    'item_to_idx': item_to_idx,
    'item_factors_norm': item_factors_norm,
    'user_factors_norm': user_factors_norm,
    'dimension': DIM,
    'n_items': N_ITEMS,
    'n_queries_benchmarked': N_QUERIES,
    'k_results': K_RESULTS,
    'nlist': nlist,
    'nprobe': nprobe,
    'exact_build_time_seconds': exact_build_time,
    'approx_build_time_seconds': approx_build_time,
    'mean_recall_at_10': mean_recall,
    'exact_latency_ms': {
        'p50': float(np.percentile(exact_latencies, 50)),
        'p75': float(np.percentile(exact_latencies, 75)),
        'p90': float(np.percentile(exact_latencies, 90)),
        'p95': float(np.percentile(exact_latencies, 95)),
        'p99': float(np.percentile(exact_latencies, 99))
    },
    'approx_latency_ms': {
        'p50': float(np.percentile(approx_latencies, 50)),
        'p75': float(np.percentile(approx_latencies, 75)),
        'p90': float(np.percentile(approx_latencies, 90)),
        'p95': float(np.percentile(approx_latencies, 95)),
        'p99': float(np.percentile(approx_latencies, 99))
    }
}

artifact_path = "data/faiss_artifacts.pkl"
temp_artifact_path = "data/faiss_artifacts_temp.pkl"

with open(temp_artifact_path, "wb") as f:
    pickle.dump(
        faiss_artifacts,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

os.replace(temp_artifact_path, artifact_path)

print("    Saved -> data/faiss_artifacts.pkl")
print("    Move to: 03_hybrid_routing.py")
