# Lab 2.2 — Item-Item Collaborative Filtering

## Purpose
This lab builds an item-item collaborative filtering recommender from implicit user interactions.

It uses:
- weighted user-item interactions
- a sparse interaction matrix
- cosine similarity between items
- Precision@10 evaluation against purchase history
- comparison with the content-based baseline from Lab 2.1

## Setup
- Python 3.11+
- Required packages from `requirements.txt`:
  - `xgboost`
  - `lightgbm`
  - `optuna`
  - `shap`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
- Input data expected under `data/`:
  - `events.csv`
  - `cb_artifacts.pkl` from Lab 2.1

## How to Run
From the lab directory, run:

```bash
python collaborative_filtering.py
```

The script builds the interaction matrix, computes item similarity, evaluates recommendations, and saves artifacts for later labs.

## Outputs
The script prints:
- total events and user-item pairs
- matrix shape and sparsity details
- sample collaborative filtering recommendations
- Precision@10 results
- comparison with the Lab 2.1 content-based baseline

It also creates:
- `output/02_cf_sparsity_analysis.png`
- `data/cf_artifacts.pkl`

## Key Design Choices
- Assigned higher weights to stronger implicit signals like `addtocart` and `transaction`.
- Built a sparse CSR user-item matrix for efficiency.
- Used item-item cosine similarity for retrieval.
- Precomputed similarities for the most popular items to reduce runtime.
- Fell back to on-the-fly similarity for items outside the precomputed set.
- Evaluated on users with multiple purchases to approximate next-item prediction.
- Saved artifacts for reuse in later recommender labs.

## Key Findings
Typical outcomes from this lab include:
- Collaborative filtering can capture co-purchase patterns that content-based methods miss.
- Sparse interaction data creates a long-tail popularity problem.
- Popular items are easier to recommend accurately.
- Precision@10 can be compared directly with the Lab 2.1 baseline.

## Extra Info
- The script suppresses warnings for cleaner output.
- The `output/` and `data/` folders should exist before running the script.
- Run Lab 2.1 first so `cb_artifacts.pkl` is available.
- The script prints a note to continue with `03_lightfm_cold_start.py`.
