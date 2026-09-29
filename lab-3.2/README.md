# Lab 3.2 — FAISS Index for ALS Embeddings

## Purpose
This lab builds and benchmarks a FAISS vector index over ALS embeddings to support low-latency nearest-neighbor retrieval.

It focuses on:
- normalized ALS item and user embeddings
- exact vs approximate FAISS search
- latency benchmarking
- ANN recall against exact search
- sample recommendations from the index

## Setup
- Python 3.11+
- Required packages from `requirements.txt`:
  - `faiss-cpu`
  - `joblib`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scipy`
  - plus the shared ML packages listed in the file
- Input data expected under `data/`:
  - `als_artifacts.pkl` from Lab 3.1

## How to Run
From the lab directory, run:

```bash
python faiss_index.py
```

The script loads ALS embeddings, builds FAISS indices, benchmarks retrieval latency, and compares approximate search with exact search.

## Outputs
The script prints:
- ALS artifact details
- index build times
- latency percentiles for exact and approximate search
- ANN recall vs exact search
- sample recommendations for a few users

It also creates:
- `output/03_faiss_latency_analysis.png`
- any saved index artifacts produced by the script for later labs

## Key Design Choices
- Normalized embeddings so inner product matches cosine similarity.
- Built both exact and approximate FAISS indices.
- Used a candidate subset to keep the demo fast.
- Benchmarked p50 and p99 latency to reflect real retrieval performance.
- Compared ANN results against exact search to measure quality loss.
- Included a NumPy fallback when FAISS is unavailable.

## Key Findings
Typical outcomes from this lab include:
- FAISS can dramatically reduce retrieval latency.
- Approximate search usually trades a small amount of recall for speed.
- Exact search is more accurate but slower.
- The 20ms target helps judge whether the index is production-ready.

## Extra Info
- The script suppresses warnings for cleaner output.
- If FAISS is not installed, the script falls back to brute-force NumPy search.
- Run Lab 3.1 first so `als_artifacts.pkl` is available.
- The `output/` and `data/` folders should exist before running the script.
