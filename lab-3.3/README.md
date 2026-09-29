# Lab 3.3 — Hybrid Routing Layer

## Purpose
This lab builds a production-style routing layer that combines multiple recommenders and post-processing steps.

It includes:
- ALS routing for returning users
- LightFM routing for cold-start users
- already-purchased exclusion
- MMR diversity re-ranking
- latency-aware recommendation assembly

## Setup
- Python 3.11+
- Required packages from `requirements.txt`:
  - `lightfm-next`
  - `faiss-cpu`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scipy`
  - plus the shared ML packages listed in the file
- Input data expected under `data/`:
  - `events.csv`
  - `routing_split.pkl` from the routing step
  - `lightfm_artifacts.pkl` from Lab 2.3
  - `als_artifacts.pkl` from Lab 3.1
  - `faiss_artifacts.pkl` from Lab 3.2

## How to Run
From the lab directory, run:

```bash
python hybrid_routing.py
```

The script loads all prerequisite artifacts, applies routing rules, filters and reranks recommendations, and prints sample outputs.

## Outputs
The script prints:
- artifact loading status
- routing eligibility details
- engine selection results
- recommendation samples
- latency information for routed requests
- summary metrics for the hybrid pipeline

It also creates:
- plots in `output/`
- any saved routing artifacts produced by the script for later labs

## Key Design Choices
- Routed users to ALS when they had enough historical interactions.
- Routed sparse or new users to LightFM for cold-start handling.
- Excluded already-purchased items from final recommendations.
- Applied MMR to improve diversity in the final list.
- Used a candidate multiplier to preserve filtering room before reranking.
- Kept the routing logic modular so it can be reused in production.

## Key Findings
Typical outcomes from this lab include:
- Hybrid routing improves recommendation robustness across user types.
- ALS works best for returning users with enough history.
- LightFM is more suitable for sparse or new users.
- Diversity and exclusion rules improve recommendation quality.

## Extra Info
- The script suppresses warnings for cleaner output.
- Run Labs 2.3, 3.1, and 3.2 first so all artifacts are available.
- The `output/` and `data/` folders should exist before running the script.
- This lab is the bridge between model training and production-style serving.
