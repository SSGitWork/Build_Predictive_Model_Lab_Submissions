# Lab 2.3 — LightFM Hybrid Recommender

## Purpose
This lab builds a hybrid recommender with LightFM and compares it against the collaborative filtering baseline from Lab 2.2.

It uses:
- WARP loss for ranking optimization
- user-item interaction data
- item feature embeddings from product metadata
- a temporal train/test split
- Precision@K, Recall@K, and AUC evaluation

## Setup
- Python 3.11+
- Required packages from `requirements.txt`:
  - `lightfm-next`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scipy`
  - plus the shared ML packages listed in the file
- Input data expected under `data/`:
  - `events.csv`
  - `item_properties_part1.csv`
  - `item_properties_part2.csv`
  - `cf_artifacts.pkl` from Lab 2.2

## How to Run
From the lab directory, run:

```bash
python lightfm_coldstart.py
```

The script loads the previous lab artifacts, builds item features, creates train/test matrices, trains LightFM models, and evaluates cold-start improvement.

## Outputs
The script prints:
- dataset and feature counts
- train/test split details
- evaluation sample size
- Pure CF training progress
- hybrid model training progress
- Precision@K / Recall@K / AUC results
- comparison against the Lab 2.2 baseline

It also creates:
- plots in `output/`
- updated artifacts for downstream use, if enabled by the script

## Key Design Choices
- Reused the collaborative filtering artifact from Lab 2.2 as the baseline reference.
- Built item features from property-value pairs.
- Used a temporal split to better simulate future recommendation behavior.
- Filtered test interactions to ensure they were unseen during training.
- Sampled evaluation users to keep runtime manageable.
- Trained LightFM incrementally so progress checkpoints could be measured.
- Compared pure CF against the hybrid feature-aware model.

## Key Findings
Typical outcomes from this lab include:
- Hybrid LightFM usually improves cold-start performance over pure collaborative filtering.
- Item features help when interaction history is sparse.
- Temporal evaluation is more realistic than a random split.
- Ranking metrics provide a better view of recommender quality than accuracy alone.

## Extra Info
- The script suppresses warnings for cleaner output.
- The default configuration uses 10 epochs; the comment notes that 20 epochs is the full prescribed setup.
- Run Lab 2.2 first so `cf_artifacts.pkl` is available.
- The `output/` and `data/` folders should exist before running the script.
