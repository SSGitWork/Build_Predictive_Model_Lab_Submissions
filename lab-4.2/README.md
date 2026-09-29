# Lab 4.2 — Recommendation Quality Monitoring

## Purpose
This lab builds a monitoring workflow for recommender quality and simulates catalog change.

It tracks:
- catalog coverage
- novelty
- intra-list diversity
- degradation after adding new items

## Setup
- Python 3.11+
- Required packages from `requirements.txt`:
  - `fastapi`
  - `uvicorn`
  - `redis`
  - `mlflow`
  - `pytest`
  - `evidently`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - plus the shared ML packages listed in the file
- Input artifacts expected under `data/`:
  - `als_artifacts.pkl`
  - `lightfm_serving.pkl`
  - `faiss_artifacts.pkl`
  - `events.csv`

## How to Run
From the lab directory, run:

```bash
python ml_monitoring.py
```

The script loads recommender artifacts, computes baseline quality metrics, simulates catalog growth, and compares the before/after results.

## Outputs
The script prints:
- artifact loading status
- baseline coverage, novelty, and diversity
- extended-catalog metrics after adding new items
- degradation or improvement signals
- alert-related summary values

It also creates:
- monitoring plots in `output/`
- any saved monitoring artifacts produced by the script for later labs

## Key Design Choices
- Used coverage, novelty, and diversity as the core quality signals.
- Measured baseline recommendations on a sample of users.
- Simulated new catalog items with random unit embeddings.
- Compared baseline and extended-catalog behavior.
- Used a tolerance threshold to flag meaningful metric changes.
- Kept the monitoring logic independent from model training.

## Key Findings
Typical outcomes from this lab include:
- Coverage usually changes when the catalog expands.
- Novelty can increase when new items are introduced.
- Diversity helps detect overly similar recommendation lists.
- Monitoring metrics are useful for spotting recommender drift.

## Extra Info
- The script suppresses warnings for cleaner output.
- Run the prerequisite recommender labs first so all artifacts are available.
- The `output/` and `data/` folders should exist before running the script.
- This lab is intended as the monitoring layer for the recommender system.
