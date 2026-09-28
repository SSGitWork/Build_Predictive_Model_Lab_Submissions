# Lab 2.1 — Content-Based Recommender

## Purpose
This lab builds a content-based product recommender using item metadata.

It uses:
- TF-IDF to convert item properties into item vectors
- cosine similarity to retrieve similar items
- a simple Precision@10 evaluation using purchase history

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
  - `item_properties_part1.csv`
  - `item_properties_part2.csv`
  - `events.csv`

## How to Run
From the lab directory, run:

```bash
python content_based_recommender.py
```

The script builds item documents, creates a TF-IDF matrix, tests recommendations, evaluates Precision@10, and saves artifacts for later labs.

## Outputs
The script prints:
- item property and document counts
- TF-IDF matrix shape and vocabulary size
- sample recommendations for one item
- Precision@10 evaluation results
- summary of evaluated users and hits

It also creates:
- `output/01_content_based_analysis.png`
- `data/cb_artifacts.pkl`

## Key Design Choices
- Combined both item property files into one catalog table.
- Kept the latest property snapshot per item-property pair.
- Converted property values into normalized text tokens.
- Used TF-IDF with bigrams and frequency filtering to build item profiles.
- Used cosine similarity for nearest-neighbor retrieval.
- Evaluated on users with at least two purchases to create a simple next-item test.
- Saved artifacts for reuse in later recommender labs.

## Key Findings
Typical outcomes from this lab include:
- Item metadata can support meaningful similarity-based recommendations.
- TF-IDF captures useful structure from product properties.
- Precision@10 provides a basic quality check for recommendation relevance.
- The recommender is easy to interpret and fast to compute.

## Extra Info
- The script suppresses warnings for cleaner output.
- The `output/` and `data/` folders should exist before running the script.
- The saved pickle file is intended for downstream recommender labs.
- The script prints a note to continue with `02_collaborative_filtering.py`.
