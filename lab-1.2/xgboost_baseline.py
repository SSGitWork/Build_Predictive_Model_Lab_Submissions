import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, RocCurveDisplay, PrecisionRecallDisplay
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  MODULE 1 | LAB 1.2")
print("  XGBoost Baseline vs Logistic Regression")
print("  Features: RAW counts only (no engineering yet)")
print("=" * 60)

print("\n[1] Loading events and building minimal feature matrix...")

events = pd.read_csv("data/events.csv")
events['datetime'] = pd.to_datetime(events['timestamp'], unit='ms')

# Target Definition: identifying users who ever made a purchase
# TODO: Extract a unique set or list of 'visitorid's whose 'event' type equals 'transaction'
purchaser_set = set(
    events.loc[events['event'] == 'transaction', 'visitorid'].unique()
)

# TODO: Filter the events dataframe to create a copy containing only actions leading up to purchases ('view', 'addtocart')
pre_purchase = events[events['event'].isin(['view', 'addtocart'])].copy()


# --- EXTRACTION OF RAW COUNTS ---

# TODO: Calculate total views per user. Filter 'pre_purchase' for 'view' events, group by 'visitorid', get sizes, and reset index.
total_views = (
    pre_purchase[pre_purchase['event'] == 'view']
    .groupby('visitorid')
    .size()
    .reset_index(name='total_views')
)

# TODO: Calculate total add-to-cart operations per user. Group by 'visitorid', extract size, reset index.
total_addtocart = (
    pre_purchase[pre_purchase['event'] == 'addtocart']
    .groupby('visitorid')
    .size()
    .reset_index(name='total_addtocart')
)

# TODO: Calculate the number of unique items viewed by each user.
# Hint: Group by 'visitorid', pull the 'itemid' column, and compute number of unique entries (.nunique())
unique_items = (
    pre_purchase[pre_purchase['event'] == 'view']
    .groupby('visitorid')['itemid']
    .nunique()
    .reset_index(name='unique_items_viewed')
)

# TODO: Count unique active days per user. Group by 'visitorid' and apply a lambda to count unique dates extracted from 'datetime'.
active_days = (
    pre_purchase
    .assign(event_date=pre_purchase['datetime'].dt.normalize())
    .groupby('visitorid')['event_date']
    .nunique()
    .reset_index(name='active_days')
)

# --- MERGING & ALIGNMENT ---

# TODO: Build a baseline feature matrix matching all unique users
all_users = events['visitorid'].unique()
user_df = pd.DataFrame({'visitorid': all_users})

# TODO: Merge total_views, total_addtocart, unique_items, and active_days into user_df using a 'left' join on 'visitorid'
# Fill any resulting missing entries (NaN) with 0 since absence implies 0 occurrences.
user_df = (
    user_df
    .merge(total_views, on='visitorid', how='left')
    .merge(total_addtocart, on='visitorid', how='left')
    .merge(unique_items, on='visitorid', how='left')
    .merge(active_days, on='visitorid', how='left')
    .fillna(0)
)

# TODO: Create the binary label 'purchased' (1 if visitorid exists inside purchaser_set, else 0)
user_df['purchased'] = user_df['visitorid'].isin(purchaser_set).astype(int)

RAW_FEATURES = [
    'total_views',
    'total_addtocart',
    'unique_items_viewed',
    'active_days'
]

print(f"\n[2] Baseline feature matrix:")
print(f"    Features used   : {RAW_FEATURES}")
print(f"    Number of features : {len(RAW_FEATURES)}")
print(f"    Total users     : {len(user_df):,}")
print(f"    Purchase rate   : {user_df['purchased'].mean()*100:.2f}%")
print(f"\n    Sample statistics:")
print(user_df[RAW_FEATURES].describe().round(2).to_string())

# Save matrix state for Lab 1.3
user_df.to_csv("data/user_features_baseline.csv", index=False)
print(f"\n    Saved -> data/user_features_baseline.csv")


# ==========================================
# [3] SPLITTING & IMBLANCE WEIGHTING
# ==========================================
print("\n[3] Splitting data (80/20 stratified)...")

# TODO: Assign the feature columns (RAW_FEATURES) to X and the label column ('purchased') to y
X = user_df[RAW_FEATURES]
y = user_df['purchased']

# TODO: Split X and y into train and test splits (80/20 ratio, set random_state to 42, stratify across target y)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# TODO: Compute negative-to-positive class ratio to manage structural dataset imbalance for tree algorithms
# Hint: Calculate total negative records (y_train == 0) divided by total positive records (y_train == 1)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

print(f"    Train : {len(X_train):,}  |  Test : {len(X_test):,}")
print(f"    Positives in train: {(y_train == 1).sum():,}  |  scale_pos_weight: {scale_pos_weight:.1f}")


# ==========================================
# [4] LOGISTIC REGRESSION TRAIN & EVAL
# ==========================================
print("\n[4] Training Logistic Regression baseline...")

# TODO: Instantiate and fit a StandardScaler on training inputs, then transform both training and testing partitions
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# TODO: Initialize LogisticRegression setting class_weight='balanced', max_iter=1000, random_state=42 and fit using scaled data
lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
lr_model.fit(X_train_scaled, y_train)

# TODO: Predict probabilities for the positive class on X_test_scaled
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

# TODO: Compute ROC-AUC and Average Precision (PR-AUC) metrics for the linear model
lr_auc = roc_auc_score(y_test, lr_proba)
lr_ap = average_precision_score(y_test, lr_proba)

print(f"    Logistic Regression AUC-ROC : {lr_auc:.4f}")
print(f"    Logistic Regression Avg-PR  : {lr_ap:.4f}")


# ==========================================
# [5] XGBOOST TRAIN & EVAL
# ==========================================
print("\n[5] Training XGBoost baseline (raw features only)...")

# TODO: Initialize an xgb.XGBClassifier model instance.
# Hyperparameters: n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
# scale_pos_weight=scale_pos_weight, eval_metric='auc', early_stopping_rounds=20, random_state=42
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    early_stopping_rounds=20,
    random_state=42
)

# TODO: Fit the XGBoost model utilizing X_train and y_train.
# Provide eval_set=[(X_test, y_test)] to allow early stopping monitor evaluation, with verbose=False.
# Note: Tree models handle unscaled raw counts natively.
xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# TODO: Capture positive class predictive probabilities on X_test, then evaluate ROC-AUC and Average Precision
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_proba)
xgb_ap = average_precision_score(y_test, xgb_proba)

print(f"    XGBoost AUC-ROC : {xgb_auc:.4f}")
print(f"    XGBoost Avg-PR  : {xgb_ap:.4f}")
print(f"    Best iteration  : {xgb_model.best_iteration if xgb_model is not None else 'N/A'}")


# ==========================================
# [6] PERFORMANCE COMPARISON MATRIX
# ==========================================
print("\n[6] Baseline comparison:")
print(f"\n    {'Model':<30} {'AUC-ROC':>10} {'Avg-PR':>10}")
print(f"    {'-'*50}")
print(f"    {'Logistic Regression':<30} {lr_auc:>10.4f} {lr_ap:>10.4f}")
print(f"    {'XGBoost (raw counts)':<30} {xgb_auc:>10.4f} {xgb_ap:>10.4f}")
print(f"    {'Difference (XGB - LR)':<30} {xgb_auc - lr_auc:>+10.4f} {xgb_ap - lr_ap:>+10.4f}")
# Ensure you evaluate differences by computing: (xgb_auc - lr_auc) and (xgb_ap - lr_ap)


# ==========================================
# [7] EVALUATION PLOTTING PIPELINE
# ==========================================
print("\n[7] Plotting evaluation curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Lab 1.2: Baseline Models — Raw Features Only\n(No feature engineering yet)",
    fontsize=13,
    fontweight='bold'
)

# --- ROC Curve Construction ---
# TODO: Use RocCurveDisplay.from_predictions to overlay both the Logistic Regression and XGBoost prediction curves onto axes[0]
# Hint: Set ax=axes[0] and use clear naming arguments for your legend labels
RocCurveDisplay.from_predictions(
    y_test,
    lr_proba,
    name=f"Logistic Regression (AUC={lr_auc:.3f})",
    ax=axes[0]
)

RocCurveDisplay.from_predictions(
    y_test,
    xgb_proba,
    name=f"XGBoost (AUC={xgb_auc:.3f})",
    ax=axes[0]
)

# TODO: Draw a baseline random reference dashed line from (0,0) to (1,1) using axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0].set_title("ROC Curve — Baseline Models")
axes[0].legend(fontsize=9)

# --- Precision-Recall Curve Construction ---
# TODO: Use PrecisionRecallDisplay.from_predictions to display both model performances on axes[1]
# Hint: Pass ax=axes[1] and label each using calculated average precision (AP) scores
PrecisionRecallDisplay.from_predictions(
    y_test,
    lr_proba,
    name=f"Logistic Regression (AP={lr_ap:.3f})",
    ax=axes[1]
)

PrecisionRecallDisplay.from_predictions(
    y_test,
    xgb_proba,
    name=f"XGBoost (AP={xgb_ap:.3f})",
    ax=axes[1]
)

axes[1].set_title("Precision-Recall — Baseline Models")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig("output/02_baseline_evaluation.png", dpi=150, bbox_inches='tight')
plt.show()


# ==========================================
# [8] TREE IMPORTANCE METRIC ANALYSIS
# ==========================================
print("\n[8] XGBoost feature importance (raw features)...")

# TODO: Construct a feature importance DataFrame containing columns 'feature' (RAW_FEATURES) and 'importance' (xgb_model.feature_importances_)
# Sort it descending by importance score
imp_df = pd.DataFrame({
    'feature': RAW_FEATURES,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(imp_df.to_string(index=False) if imp_df is not None else "    Not Implemented")

# --- Horizontal Importance Plotting ---
fig, ax = plt.subplots(figsize=(7, 4))
# TODO: Render an ax.barh layout tracing features against metrics.
# Hint: Reversing the order using slice indexing [::-1] helps visually display the top performers at the apex of the plot.
ax.barh(
    imp_df['feature'][::-1],
    imp_df['importance'][::-1],
    color='#4C72B0',
    edgecolor='white'
)

ax.set_title("Feature Importance — Raw Features Only\n(Before engineering)", fontweight='bold')
ax.set_xlabel("Importance (Gain)")
plt.tight_layout()
plt.savefig("output/02_baseline_feature_importance.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("  LAB 1.2 COMPLETE — BOOKMARK THESE NUMBERS")
print("=" * 60)
