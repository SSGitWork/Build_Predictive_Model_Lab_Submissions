# Lab 1.2 — Baseline Models for Purchase Prediction

## Purpose
This lab builds and compares two baseline classifiers for purchase prediction using raw user activity counts:
- Logistic Regression
- XGBoost

The goal is to establish a simple benchmark before feature engineering and model tuning.

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

## How to Run
From the lab directory, run:

```bash
python xgboost_baseline.py
```

The script loads the event data, creates a minimal feature matrix, trains both models, evaluates them, and saves the feature table for the next lab.

## Outputs
The script prints:
- feature matrix summary
- train/test split details
- class imbalance weight
- ROC-AUC and Average Precision for both models
- comparison table of model performance

It also creates:
- `data/user_features_baseline.csv`
- an evaluation plot showing ROC and Precision-Recall curves

## Key Design Choices
- Used only raw count features to create a true baseline.
- Defined the target as whether a user ever completed a `transaction`.
- Built features from pre-purchase behavior only (`view`, `addtocart`).
- Applied `StandardScaler` for Logistic Regression, but not for XGBoost.
- Used `class_weight='balanced'` for Logistic Regression and `scale_pos_weight` for XGBoost to handle class imbalance.
- Used stratified train/test splitting to preserve the purchase ratio.
- Enabled early stopping for XGBoost.

## Key Findings
Typical outcomes from this lab include:
- Purchase prediction is highly imbalanced.
- XGBoost usually performs better than Logistic Regression on raw behavioral counts.
- Users with more views, add-to-cart actions, and active days are more likely to purchase.
- Raw features provide a useful baseline, but feature engineering is likely needed for stronger performance.

## Extra Info
- The script suppresses warnings for cleaner output.
- The generated baseline feature file is intended for use in Lab 1.3.
- If the `data/` or `output/` folders are missing, create them before running the script.
