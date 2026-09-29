# Lab 3.4 — Offline A/B Evaluation Protocol

## Purpose
This lab simulates an offline A/B test for the recommender system and evaluates whether the hybrid engine outperforms the baseline.

It covers:
- deterministic control/treatment assignment
- sample size estimation
- NDCG@10 evaluation
- hypothesis testing
- guardrail and comparison visualizations

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
  - `routing_split.pkl`
  - `als_artifacts.pkl`
  - `lightfm_artifacts.pkl`
  - `routing_artifacts.pkl`

## How to Run
From the lab directory, run:

```bash
python ab_testing.py
```

The script loads the prerequisite artifacts, assigns users to control and treatment groups, simulates offline A/B evaluation, and runs statistical tests.

## Outputs
The script prints:
- user group counts
- sample size estimates
- control vs treatment NDCG@10
- relative lift and p-value
- significance decision
- progress through the evaluation loop

It also creates:
- plots in `output/`
- any saved A/B protocol artifacts produced by the script for later labs

## Key Design Choices
- Used deterministic hashing for sticky group assignment.
- Estimated sample size from a baseline NDCG@10 and a relative MDE.
- Evaluated only users with future purchases.
- Routed treatment users to ALS when they had enough history.
- Used LightFM as the fallback engine for the control path.
- Applied a one-tailed t-test for the directional hypothesis.
- Added plots to make the A/B result easier to interpret.

## Key Findings
Typical outcomes from this lab include:
- The hybrid treatment can outperform the baseline control.
- Sample size depends heavily on the baseline metric and MDE.
- Deterministic assignment makes offline experiments reproducible.
- Statistical significance should be interpreted alongside practical lift.

## Extra Info
- The script suppresses warnings for cleaner output.
- Run the prerequisite labs first so all artifacts are available.
- The `output/` and `data/` folders should exist before running the script.
- This lab is the final evaluation step for the recommender workflow.
