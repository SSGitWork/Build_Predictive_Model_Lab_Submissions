# =============================================================================
# MODULE 2 | LAB 2.1
# File: 01_content_based_recommender.py
# Purpose: Build a content-based product recommender using TF-IDF on
#          item properties and cosine similarity retrieval
# Saras AI Institute | Build Predictive Models & Modern Recommenders
# =============================================================================

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import scipy.sparse as sp
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("  MODULE 2 | LAB 2.1")
print("  Content-Based Recommender")
print("  Method: TF-IDF + Cosine Similarity")
print("=" * 60)

# ---------------------------------------------------------------------------
# SECTION 1: Load Item Properties
# ---------------------------------------------------------------------------
print("\n[1] Loading item properties...")

props1 = pd.read_csv("data/item_properties_part1.csv")
props2 = pd.read_csv("data/item_properties_part2.csv")

# TODO: Combine props1 and props2 vertically into a unified dataframe named 'props'
props = pd.concat([props1, props2], ignore_index=True)

print(f"    Total property records : {len(props):,}")
print(f"    Unique items           : {props['itemid'].nunique():,}")
print(f"    Columns                : {list(props.columns)}")


# ---------------------------------------------------------------------------
# SECTION 2: Build Item Documents
# ---------------------------------------------------------------------------
print("\n[2] Building item documents from properties...")

# TODO: Deduplicate properties to keep only the most recent snapshot of each item-property pair.
# Hint: Sort 'props' by 'timestamp' descending, then use drop_duplicates across ['itemid', 'property']
props_latest = (
    props
    .sort_values('timestamp', ascending=False)
    .drop_duplicates(subset=['itemid', 'property'], keep='first')
    .copy()
)

# TODO: Transform property values into tokens and concatenate them per item into a single string document
# Hint: Group 'props_latest' by 'itemid', extract the 'value' column, and apply a lambda or string
# transformation to lowercase values, replace spaces with underscores, and ' '.join() them.
props_latest['value_token'] = (
    props_latest['value']
    .fillna('')
    .astype(str)
    .str.lower()
    .str.replace(r'\s+', '_', regex=True)
)

item_docs = (
    props_latest
    .groupby('itemid')['value_token']
    .apply(lambda values: ' '.join(token for token in values if token))
    .reset_index(name='document')
)

# Remove any items which may have no usable property-value tokens
item_docs = item_docs[item_docs['document'].str.strip() != ''].reset_index(drop=True)

print(f"    Items with documents   : {len(item_docs):,}")


# ---------------------------------------------------------------------------
# SECTION 3: Build TF-IDF Matrix
# ---------------------------------------------------------------------------
print("\n[3] Building TF-IDF matrix...")

# TODO: Instantiate a TfidfVectorizer object to convert item text documents into numerical profiles.
# Hyperparameters: max_features=5000, min_df=2, max_df=0.95, ngram_range=(1, 2), sublinear_tf=True
vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 2),
    sublinear_tf=True
)

# TODO: Fit the vectorizer on the item_docs['document'] column and transform the text into a sparse TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(item_docs['document'])

# TODO: Extract the raw array of unique item IDs from item_docs['itemid'] for index-to-ID mappings
item_ids = item_docs['itemid'].values

print(f"    TF-IDF matrix shape    : {tfidf_matrix.shape}")
print(f"    Vocabulary size        : {len(vectorizer.vocabulary_):,} terms")


# ---------------------------------------------------------------------------
# SECTION 4: Build the Recommendation Function
# ---------------------------------------------------------------------------
print("\n[4] Building recommendation function...")

def get_similar_items(item_id, tfidf_matrix, item_ids, top_k=10):
    """
    Given an item_id, return the top_k most similar items based on TF-IDF cosine similarity.
    Returns a pandas DataFrame containing 'item_id' and 'similarity' columns.
    """

    # TODO: Verify if item_id exists in our catalog array (item_ids). If not, print warning and return empty DataFrame.
    if item_id not in item_ids:
        print(f"    Warning: Item ID {item_id} was not found in the catalog.")
        return pd.DataFrame(columns=['item_id', 'similarity'])

    # TODO: Extract the exact row coordinate index matching the query item_id inside your item_ids list
    idx = np.where(item_ids == item_id)[0][0]

    # TODO: Isolate the row vector from tfidf_matrix at index 'idx'
    item_vector = tfidf_matrix[idx]

    # TODO: Calculate the cosine similarity matrix between the isolated item_vector and the full tfidf_matrix.
    # Hint: Use cosine_similarity(item_vector, tfidf_matrix) and flatten the resulting 2D output array into a 1D array.
    similarities = cosine_similarity(item_vector, tfidf_matrix).flatten()

    # TODO: Sort similarities in descending order and identify indices of the top_k most similar items.
    # Note: Exclude the first item itself (as similarity with itself equals 1.0) by slicing [1:top_k+1]
    top_indices = np.argsort(similarities)[::-1][1:top_k + 1]

    # TODO: Build and return a pandas DataFrame mapping calculated values to 'item_id' and 'similarity' keys
    results = pd.DataFrame({
        'item_id': item_ids[top_indices],
        'similarity': similarities[top_indices]
    })

    return results


# ---------------------------------------------------------------------------
# SECTION 5: Test the Recommender
# ---------------------------------------------------------------------------
print("\n[5] Testing content-based recommender...")

if len(item_ids) > 0 and tfidf_matrix is not None:
    sample_index = min(100, len(item_ids) - 1)
    sample_item = item_ids[sample_index]

    print(f"    Query item ID: {sample_item}")

    # TODO: Call your get_similar_items function to find the top 10 matches for sample_item
    recommendations = get_similar_items(
        sample_item,
        tfidf_matrix,
        item_ids,
        top_k=10
    )

    print(recommendations.to_string(index=False))


# ---------------------------------------------------------------------------
# SECTION 6: Evaluate — Precision@K for Purchased Items
# ---------------------------------------------------------------------------
print("\n[6] Evaluating with purchase data...")

events = pd.read_csv("data/events.csv")

# TODO: Filter events down to 'transaction' occurrences only and pull out ['visitorid', 'itemid']
purchases = (
    events[events['event'] == 'transaction'][['visitorid', 'itemid', 'timestamp']]
    .sort_values(['visitorid', 'timestamp'])
    .copy()
)

purchases.columns = ['user_id', 'item_id', 'timestamp']

# TODO: Isolate sequential multi-purchase records by keeping users with 2 or more historical purchases
multi_buyers = purchases[
    purchases.groupby('user_id')['item_id'].transform('size') >= 2
].copy()

# Evaluation loops over 500 multi-purchase buyers
hits_at_10 = 0
total_eval = 0

if len(multi_buyers) > 0:
    eval_users = multi_buyers['user_id'].unique()[:500]

    for user in eval_users:
        user_items = purchases[purchases['user_id'] == user]['item_id'].values

        # Filter out user interactions that don't exist in our item metadata catalog boundaries
        catalog_items = [i for i in user_items if i in item_ids]

        if len(catalog_items) < 2:
            continue

        query_item = catalog_items[0]
        target_item = catalog_items[1]

        # TODO: Retrieve the top 10 recommended items for the query_item using your get_similar_items framework
        recs = get_similar_items(
            query_item,
            tfidf_matrix,
            item_ids,
            top_k=10
        )

        if recs is None or len(recs) == 0:
            continue

        # TODO: Check if target_item exists inside the recommended item list. If yes, increment hits_at_10.
        # Increment total_eval counters for each user evaluation run completed.
        if target_item in recs['item_id'].values:
            hits_at_10 += 1

        total_eval += 1

# TODO: Calculate precision_at_10 by dividing hits_at_10 by total_eval
precision_at_10 = hits_at_10 / total_eval if total_eval > 0 else 0.0

print(f"\n    Evaluated on {total_eval} users")
print(f"    Hits@10     : {hits_at_10}")
print(f"    Precision@10: {precision_at_10:.4f}")

cb_results = {
    'model': 'Content-Based (TF-IDF)',
    'precision_at_10': precision_at_10,
    'evaluated_users': total_eval
}


# ---------------------------------------------------------------------------
# SECTION 7: Similarity Distribution Visualization
# ---------------------------------------------------------------------------
print("\n[7] Visualizing similarity distribution...")

os.makedirs("output", exist_ok=True)

# --- Plotting Baseline Profiles ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Lab 2.1: Content-Based Recommender — TF-IDF Analysis",
    fontsize=13,
    fontweight='bold'
)

sample_index = min(100, len(item_ids) - 1)
sample_vector = tfidf_matrix[sample_index]

# Limit to 1,000 catalog items for the distribution chart
comparison_limit = min(1000, tfidf_matrix.shape[0])
all_sims = cosine_similarity(
    sample_vector,
    tfidf_matrix[:comparison_limit]
).flatten()

# TODO: Render a histogram on axes[0] illustrating raw cosine similarities calculated from an item against the catalog array
# Hint: Use axes[0].hist(all_sims, bins=50)
axes[0].hist(
    all_sims,
    bins=50,
    color='#4C72B0',
    edgecolor='white'
)

axes[0].set_title("Cosine Similarity Distribution\n(Reference item vs 1000 items)")
axes[0].set_xlabel("Cosine Similarity Score")
axes[0].set_ylabel("Number of Items")

# --- Plotting Document Feature Weights ---
# TODO: Calculate top terms for a sample document by sorting vocabulary feature weights extracted from tfidf_matrix rows
# Plot top 15 weights as a horizontal bar plot on axes[1] using ax.barh()
feature_names = np.array(vectorizer.get_feature_names_out())
sample_weights = tfidf_matrix[sample_index].toarray().flatten()

top_term_indices = np.argsort(sample_weights)[::-1][:15]
top_term_indices = top_term_indices[sample_weights[top_term_indices] > 0]

top_terms = feature_names[top_term_indices]
top_weights = sample_weights[top_term_indices]

axes[1].barh(
    top_terms[::-1],
    top_weights[::-1],
    color='#55A868'
)

axes[1].set_title("Top TF-IDF Terms\nfor Sample Item")
axes[1].set_xlabel("TF-IDF Score")

plt.tight_layout()
plt.savefig("output/01_content_based_analysis.png", dpi=150, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------------------
# SECTION 8: Save Artifacts for Lab 2.4
# ---------------------------------------------------------------------------
import pickle

# TODO: Save tfidf_matrix, item_ids, vectorizer, and cb_results dictionaries to a pickle file
# Path: "data/cb_artifacts.pkl"
cb_artifacts = {
    'tfidf_matrix': tfidf_matrix,
    'item_ids': item_ids,
    'vectorizer': vectorizer,
    'cb_results': cb_results
}

with open("data/cb_artifacts.pkl", "wb") as file:
    pickle.dump(cb_artifacts, file)

print("\n    Saved -> data/cb_artifacts.pkl")
print("    Move to: 02_collaborative_filtering.py")
