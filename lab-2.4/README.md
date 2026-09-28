# Lab 2.4 — Head-to-Head Recommender Evaluation

## Purpose
This lab compares all three recommender approaches from Module 2 on the same evaluation users and candidate items.

It measures:
- Precision@K
- NDCG@K
- Coverage

It also introduces the 20/80 A/B split used later in Module 3.

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
  - `cb_artifacts.pkl` from Lab 2.1
  - `cf_artifacts.pkl` from Lab 2.2
  - `lightfm_artifacts.pkl` from Lab 2.3

## How to Run
From the lab directory, run:

```bash
python evaluation_comparison.py
```

The script loads all model artifacts, builds a shared evaluation set, computes ranking metrics, and produces comparison plots.

## Outputs
The script prints:
- artifact loading status
- shared catalog and candidate pool sizes
- evaluation user counts
- Precision@K, NDCG@K, and Coverage results
- model comparison summary
- A/B split preparation details

It also creates:
- comparison plots in `output/`
- evaluation artifacts for downstream use, if enabled by the script

## Key Design Choices
- Evaluated all models on the same users and candidate pool for fairness.
- Restricted evaluation to items available in every model catalog.
- Used a temporal split to simulate future purchases.
- Measured multiple metrics instead of relying on Precision@K alone.
- Included coverage to capture catalog diversity.
- Built helper functions for safe artifact loading and metric calculation.
- Prepared the 20/80 split as a bridge to the next module.

## Key Findings
Typical outcomes from this lab include:
- Different recommenders excel on different metrics.
- Hybrid models often balance relevance and coverage better than pure CF.
- Content-based models can be strong for niche items.
- Coverage helps reveal whether a model over-focuses on popular items.

## Extra Info
- The script suppresses warnings for cleaner output.
- Run Labs 2.1, 2.2, and 2.3 first so all artifacts are available.
- The `output/` and `data/` folders should exist before running the script.
- This lab is intended as the final evaluation step before Module 3.
