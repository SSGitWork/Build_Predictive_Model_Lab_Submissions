# Lab 1.4 — LightGBM + Optuna Tuning

## Purpose
This lab compares a reference XGBoost model against LightGBM, then uses Optuna to tune LightGBM hyperparameters.

The goal is to study the tradeoff between model quality and training time, and to identify a strong tuned configuration for the engineered feature set.

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
  - `user_features_engineered.csv` from Lab 1.3

## How to Run
From the lab directory, run:

```bash
python optuna_tuning.py
```

The script loads the engineered feature matrix, trains baseline models, runs Optuna tuning, retrains the best LightGBM model, and generates comparison plots.

## Outputs
The script prints:
- feature count and purchase rate
- XGBoost reference metrics and training time
- LightGBM default metrics and training time
- Optuna best trial results and parameters
- tuned LightGBM metrics and training time
- a Pareto summary table

It also creates:
- a Pareto tradeoff plot in `output/`
- an Optuna history / tuning visualization in `output/`

## Key Design Choices
- Reused the engineered feature matrix from Lab 1.3.
- Compared both default and tuned LightGBM against an XGBoost reference.
- Used stratified train/test splitting to preserve class balance.
- Used `scale_pos_weight` to handle class imbalance.
- Measured training time to evaluate the quality/speed tradeoff.
- Used Optuna with cross-validation to search a broad hyperparameter space.
- Kept early stopping enabled for efficient training.

## Key Findings
Typical outcomes from this lab include:
- LightGBM can match or outperform the XGBoost reference on engineered features.
- Optuna tuning often improves AUC and Average Precision.
- Better performance may come with higher training cost.
- The Pareto plot helps identify the best balance between speed and quality.

## Extra Info
- The script suppresses warnings and Optuna logging for cleaner output.
- If `data/user_features_engineered.csv` is missing, run Lab 1.3 first.
- If the `output/` folder is missing, create it before running the script.
