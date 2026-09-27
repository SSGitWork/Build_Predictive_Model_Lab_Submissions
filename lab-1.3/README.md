# Lab 1.3 — Feature Engineering Pipeline

## Purpose
This lab extends the Lab 1.2 baseline by adding feature engineering waves and measuring the lift in purchase prediction performance.

The workflow compares:
- the raw baseline features from Lab 1.2
- ratio-based features
- datetime-based features
- interaction terms

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
  - `user_features_baseline.csv` from Lab 1.2

## How to Run
From the lab directory, run:

```bash
python feature_pipeline.py
```

The script loads the event data and the baseline feature matrix, then trains XGBoost models across multiple feature sets.

## Outputs
The script prints:
- baseline and wave-by-wave model metrics
- AUC and Average Precision for each stage
- lift relative to the Lab 1.2 baseline
- a summary table of feature growth

It also creates:
- a lift progression plot in `output/`

## Key Design Choices
- Reused the Lab 1.2 baseline feature matrix instead of rebuilding it.
- Added features in waves to isolate the effect of each feature group.
- Used a shared `train_and_evaluate()` helper to keep evaluation consistent.
- Kept the same train/test split strategy and XGBoost settings across all waves.
- Used safe ratio formulas with `+ 1` to avoid division by zero.
- Added a weekend-shopping flag to capture temporal behavior.
- Included interaction terms to model combined behavioral signals.

## Key Findings
Typical outcomes from this lab include:
- Feature engineering improves performance over the raw baseline.
- Ratio features often provide a noticeable lift.
- Datetime features may add smaller but useful signal.
- Interaction terms can further improve ranking quality.
- The final engineered feature set should outperform the Lab 1.2 baseline.

## Extra Info
- The script suppresses warnings for cleaner output.
- The lab is designed to show incremental lift, not just final accuracy.
- If `data/user_features_baseline.csv` is missing, run Lab 1.2 first.
- If the `output/` folder is missing, create it before running the script.
