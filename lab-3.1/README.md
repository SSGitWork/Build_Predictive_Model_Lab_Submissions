# Lab 3.1 — ALS Personalization Engine

## Purpose
This lab builds a personalized recommender using Alternating Least Squares (ALS) on implicit feedback.

It focuses on:
- confidence-weighted user-item interactions
- temporal train/test splitting
- NDCG@10 evaluation
- confidence-weight tuning experiments

## Setup
- Python 3.11+
- Required packages from `requirements.txt`:
  - `implicit`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scipy`
  - plus the shared ML packages listed in the file
- Input data expected under `data/`:
  - `events.csv`
  - `routing_split.pkl` from the prior module/lab

## How to Run
From the lab directory, run:

```bash
python als_personalization.py
```

The script loads the routing split, builds weighted interaction matrices, trains ALS, evaluates NDCG@10, and runs a confidence-weight experiment.

## Outputs
The script prints:
- dataset and routing split details
- subsampling statistics
- matrix shapes and interaction counts
- ALS training time and factor shapes
- NDCG@10 results
- confidence-weight experiment results
- best configuration summary

It also creates:
- plots in `output/`
- any saved artifacts produced by the script for later labs

## Key Design Choices
- Used a confidence-weighted implicit feedback matrix instead of binary interactions.
- Limited the demo to the most active users and items for speed.
- Used a temporal split to simulate future behavior.
- Removed test pairs already seen in training.
- Evaluated with NDCG@10 to measure ranking quality.
- Compared multiple confidence-weight settings to find a better ALS configuration.

## Key Findings
Typical outcomes from this lab include:
- ALS can produce strong personalized recommendations from implicit data.
- Confidence weighting has a meaningful impact on ranking quality.
- Temporal evaluation is more realistic than random splitting.
- Heavier purchase weighting often improves recommendation relevance.

## Extra Info
- The script suppresses warnings for cleaner output.
- The demo uses 10 iterations by default; the comments note the full prescribed setup may use 20.
- Run the prerequisite routing split step first so `routing_split.pkl` is available.
- The `output/` and `data/` folders should exist before running the script.
