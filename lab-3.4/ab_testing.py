# =============================================================================
# MODULE 3 | LAB 3.4
# File: 04_ab_protocol.py
# Purpose: Establish an offline A/B testing simulation protocol, compute
#          statistical power boundaries, sample sizing, and run hypothesis tests.
# Saras AI Institute | Build Predictive Models & Modern Recommenders
# =============================================================================

import os
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("  MODULE 3 | LAB 3.4")
print("  Offline A/B Evaluation Protocol")
print("  Hypothesis · Sample Size · Metrics · Guardrails")
print("=" * 60)


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
K = 10
MAX_TEST_USERS = 200
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Helper Function: Safe Artifact Loading
# ---------------------------------------------------------------------------
def load_pickle_checked(path):
    """Load a non-empty pickle artifact with useful error messages."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required artifact not found: {path}. "
            "Please run the prerequisite labs first."
        )

    if os.path.getsize(path) == 0:
        raise ValueError(
            f"Artifact is empty: {path}. "
            "Please rerun the corresponding prerequisite lab."
        )

    try:
        with open(path, "rb") as file:
            return pickle.load(file)
    except EOFError as error:
        raise ValueError(
            f"Artifact is incomplete or corrupted: {path}. "
            "Please rerun the corresponding prerequisite lab."
        ) from error


# ---------------------------------------------------------------------------
# SECTION 1: Load Artifacts
# ---------------------------------------------------------------------------
print("\n[1] Loading artifacts...")

routing = load_pickle_checked("data/routing_split.pkl")
als_art = load_pickle_checked("data/als_artifacts.pkl")
lfm_art = load_pickle_checked("data/lightfm_artifacts.pkl")
rt_art = load_pickle_checked("data/routing_artifacts.pkl")

events = pd.read_csv("data/events.csv")
events['datetime'] = pd.to_datetime(events['timestamp'], unit='ms')

purchases = events[events['event'] == 'transaction'].copy()

print(f"    Total events          : {len(events):,}")
print(f"    Purchase events       : {len(purchases):,}")


# ---------------------------------------------------------------------------
# SECTION 2: Randomization Strategy
# ---------------------------------------------------------------------------
print("\n[2] Randomization Strategy")
print("=" * 55)

def assign_group(user_id, salt="ab_v1"):
    """
    Deterministically maps a user ID into Control or Treatment buckets.
    Ensures group stickiness across runtime iterations without database storage.
    """
    # TODO: Concatenate salt and user_id into a single string token, calculate its hash string integer representation,
    # and check if hash % 100 is less than 50 to return "control", otherwise "treatment".
    token = f"{salt}_{user_id}".encode("utf-8")
    hash_value = int(hashlib.sha256(token).hexdigest(), 16)

    return "control" if hash_value % 100 < 50 else "treatment"


all_users = events['visitorid'].unique()

# TODO: Build an assignment mapping registry dictionary containing {user_id: assign_group(user_id)} pairs
groups = {
    user_id: assign_group(user_id)
    for user_id in all_users
}

control_users = [
    user_id
    for user_id, group in groups.items()
    if group == "control"
]

treatment_users = [
    user_id
    for user_id, group in groups.items()
    if group == "treatment"
]

print(f"  Total users     : {len(all_users):,}")
print(f"  Control  (50%)  : {len(control_users):,}")
print(f"  Treatment (50%) : {len(treatment_users):,}")


# ---------------------------------------------------------------------------
# SECTION 3: Sample Size Calculation
# ---------------------------------------------------------------------------
print("\n[3] Sample Size Calculation")
print("=" * 55)

# Lab 3.1 saves the selected best metric under best_ndcg_at_10.
baseline_ndcg = float(
    als_art.get(
        'best_ndcg_at_10',
        als_art.get(
            'baseline_ndcg_at_10',
            als_art.get('best_ndcg', 0.017)
        )
    )
)

# Ensure a usable value for the illustrative power calculation.
baseline_ndcg = max(baseline_ndcg, 0.01)

MDE_RELATIVE = 0.05

# TODO: Compute absolute minimum detectable effect parameter size matching your relative boundaries
MDE_ABSOLUTE = baseline_ndcg * MDE_RELATIVE

alpha = 0.05
power = 0.80

# TODO: Calculate standard statistical deviation z-score intervals for the specified alpha and power boundaries
# Hint: Use stats.norm.ppf() for upper tails -> (1 - alpha / 2) and (power)
z_alpha = stats.norm.ppf(1 - alpha / 2)
z_beta = stats.norm.ppf(power)

print(f"  Baseline NDCG@10      : {baseline_ndcg:.4f}")
print(f"  MDE (5% relative)     : {MDE_ABSOLUTE:.4f} absolute")

# TODO: Formulate variance parameters and map the sample sizing equation to isolate requirements per group
# Mathematical Formula Hint: n = [2 * (Z_alpha + Z_beta)^2 * sigma^2] / delta^2
# Where variance (sigma^2) can be approximated via Bernoulli properties: baseline * (1 - baseline)
variance = baseline_ndcg * (1 - baseline_ndcg)

n_per_group = int(
    np.ceil(
        (2 * (z_alpha + z_beta) ** 2 * variance) /
        (MDE_ABSOLUTE ** 2)
    )
)

total_required = n_per_group * 2

print(f"  Required per group  : {n_per_group:,}")
print(f"  Total required      : {total_required:,}")

# ---------------------------------------------------------------------------
# SECTION 4: NDCG@K Metric Implementation
# ---------------------------------------------------------------------------
def ndcg_at_k(recommended, relevant, k=10):
    """
    Calculates Normalized Discounted Cumulative Gain at rank position k.
    Formula: NDCG@K = DCG@K / IDCG@K
    """
    rec_k = recommended[:k]

    # TODO: Implement the Discounted Cumulative Gain calculation loop over top k item values
    # Hint: Use 1.0 / np.log2(i + 2) if item belongs to relevant set
    dcg = 0.0

    for i, item_id in enumerate(rec_k):
        if item_id in relevant:
            dcg += 1.0 / np.log2(i + 2)

    # TODO: Implement Ideal Discounted Cumulative Gain loop based on maximum possible ideal matches
    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1.0 / np.log2(i + 2)
        for i in range(ideal_hits)
    )

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# SECTION 5: Offline Metric Simulation Loop
# ---------------------------------------------------------------------------
print("\n[4] Offline A/B metric simulation...")

cutoff = pd.to_datetime(routing['cutoff_date'])

train_ev = events[events['datetime'] <= cutoff].copy()
test_purch = purchases[purchases['datetime'] > cutoff].copy()

user_ix_counts = train_ev.groupby('visitorid').size().to_dict()

# User history used to exclude previously seen items from LightFM ranking.
seen_items_by_user = (
    train_ev
    .groupby('visitorid')['itemid']
    .apply(lambda values: set(values))
    .to_dict()
)

als_user_to_idx = als_art['user_to_idx']
als_item_ids = np.asarray(als_art['item_ids'])
als_model = als_art['model']

# Prefer the training matrix rather than the full interaction matrix.
als_user_item = als_art.get(
    'train_user_item_matrix',
    als_art.get('user_item_matrix')
)

lfm_user_map, _, lfm_item_map, _ = lfm_art['dataset'].mapping()
lfm_item_ids_list = list(lfm_item_map.keys())
n_lfm_items = len(lfm_item_ids_list)

lfm_model = lfm_art['model_hybrid']
lfm_item_f = lfm_art['item_features_matrix']

rng = np.random.default_rng(RANDOM_STATE)

all_test_users = test_purch['visitorid'].unique()

test_users_with_p = rng.choice(
    all_test_users,
    size=min(MAX_TEST_USERS, len(all_test_users)),
    replace=False
)

control_ndcgs = []
treatment_ndcgs = []

control_evaluated_users = []
treatment_evaluated_users = []

print(f"    Simulating on {len(test_users_with_p)} test users...")

for count, user_id in enumerate(test_users_with_p, start=1):
    group = groups.get(user_id, 'control')

    relevant = set(
        test_purch[
            test_purch['visitorid'] == user_id
        ]['itemid'].values
    )

    if not relevant:
        continue

    n_ix = user_ix_counts.get(user_id, 0)

    # TODO: Determine if conditions permit querying the returning-user engine (ALS) for treatment records
    # Constraints: group must equal treatment, user must exist in conversion map, interactions >= 3
    use_als = (
        group == "treatment" and
        user_id in als_user_to_idx and
        n_ix >= 3
    )

    recs = []

    if use_als:
        # TODO: Execute an internal model.recommend query lookup to fetch candidate IDs for the current user index
        user_idx = als_user_to_idx[user_id]

        if user_idx < als_user_item.shape[0]:
            item_indices, _ = als_model.recommend(
                userid=user_idx,
                user_items=als_user_item[user_idx],
                N=K,
                filter_already_liked_items=True
            )

            recs = list(als_item_ids[item_indices])

    elif user_id in lfm_user_map:
        # TODO: Fall back to generating raw predictions via lfm_model.predict across item matrices
        user_idx = lfm_user_map[user_id]

        item_indices = np.arange(n_lfm_items)

        scores = lfm_model.predict(
            user_ids=np.full(n_lfm_items, user_idx),
            item_ids=item_indices,
            item_features=lfm_item_f,
            num_threads=2
        )

        # Exclude items already interacted with before the temporal cutoff.
        seen_items = seen_items_by_user.get(user_id, set())

        if seen_items:
            seen_mask = np.isin(
                np.asarray(lfm_item_ids_list),
                list(seen_items)
            )
            scores[seen_mask] = -np.inf

        top_indices = np.argsort(scores)[::-1][:K]

        recs = [
            lfm_item_ids_list[index]
            for index in top_indices
            if np.isfinite(scores[index])
        ]

    if not recs:
        continue

    # TODO: Calculate NDCG@10 scores using your implemented ndcg_at_k function,
    # and append outputs to the matching group metrics tracking array (control_ndcgs or treatment_ndcgs)
    user_ndcg = ndcg_at_k(recs, relevant, k=K)

    if group == "control":
        control_ndcgs.append(user_ndcg)
        control_evaluated_users.append(user_id)
    else:
        treatment_ndcgs.append(user_ndcg)
        treatment_evaluated_users.append(user_id)

    if count % 25 == 0 or count == len(test_users_with_p):
        print(f"    Processed {count}/{len(test_users_with_p)} users...")


print(f"    Control observations   : {len(control_ndcgs):,}")
print(f"    Treatment observations : {len(treatment_ndcgs):,}")


# ---------------------------------------------------------------------------
# SECTION 6: Statistical Test
# ---------------------------------------------------------------------------
print("\n[5] Statistical Test Results")
print("=" * 55)

if control_ndcgs and treatment_ndcgs:
    # TODO: Extract aggregate distribution statistics across simulation parameters
    ctrl_mean = float(np.mean(control_ndcgs))
    treat_mean = float(np.mean(treatment_ndcgs))

    # TODO: Compute relative lift metric ratios between treatment and control means
    diff = treat_mean - ctrl_mean

    rel_lift = (
        (diff / ctrl_mean) * 100
        if ctrl_mean > 0
        else 0.0
    )

    # TODO: Execute an independent two-sample t-test across metrics containers using stats.ttest_ind()
    # Pull out structural stats indicators and calculate the one-tailed p-value
    t_stat, p_two = stats.ttest_ind(
        treatment_ndcgs,
        control_ndcgs,
        equal_var=False
    )

    # One-tailed p-value assumes the directional hypothesis:
    # Treatment NDCG@10 > Control NDCG@10.
    if np.isnan(t_stat) or np.isnan(p_two):
        p_one = 1.0
    elif t_stat > 0:
        p_one = p_two / 2
    else:
        p_one = 1 - (p_two / 2)

    statistically_significant = (
        p_one < alpha and
        treat_mean > ctrl_mean
    )

    print(f"  Control  (LightFM-only)  : NDCG@10 = {ctrl_mean:.4f}")
    print(f"  Treatment (Hybrid Engine): NDCG@10 = {treat_mean:.4f}")
    print(f"  Absolute difference      : {diff:+.4f}")
    print(f"  Relative lift            : {rel_lift:+.2f}%")
    print(f"  t-statistic              : {t_stat:.4f}")
    print(f"  p-value (one-tailed)     : {p_one:.4f}")
    print(
        f"  Decision at alpha={alpha:.2f}   : "
        f"{'Reject H0 — treatment wins' if statistically_significant else 'Do not reject H0'}"
    )

else:
    ctrl_mean = 0.0
    treat_mean = 0.0
    diff = 0.0
    rel_lift = 0.0
    t_stat = 0.0
    p_one = 1.0
    statistically_significant = False

    print("  Insufficient valid observations for a statistical comparison.")


# ---------------------------------------------------------------------------
# SECTION 7: Visualizations
# ---------------------------------------------------------------------------
print("\n[6] Plotting A/B protocol results...")

os.makedirs("output", exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

fig.suptitle(
    "Lab 3.4: Offline A/B Evaluation Protocol",
    fontsize=12,
    fontweight='bold'
)

# --- Plot 1: NDCG Distribution Contrast ---
# TODO: Build an overlaid histogram layout charting control vs treatment NDCG frequencies on axes[0]
if control_ndcgs:
    axes[0].hist(
        control_ndcgs,
        bins=15,
        alpha=0.65,
        color='#4C72B0',
        label='Control: LightFM-only'
    )

if treatment_ndcgs:
    axes[0].hist(
        treatment_ndcgs,
        bins=15,
        alpha=0.65,
        color='#55A868',
        label='Treatment: Hybrid'
    )

axes[0].set_title("NDCG@10 Distribution")
axes[0].set_xlabel("NDCG@10")
axes[0].set_ylabel("Number of Users")
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# --- Plot 2: Sample Sizing Function Curvatures ---
# TODO: Plot the generated sample size requirements progression line across the linear mde_range on axes[1]
# Draw tracking indicator baseline markers tracking configured MDE boundaries using axes[1].axvline()
mde_range = np.linspace(0.01, 0.20, 100)

sample_size_curve = (
    2 * (z_alpha + z_beta) ** 2 * variance
) / ((baseline_ndcg * mde_range) ** 2)

axes[1].plot(
    mde_range * 100,
    sample_size_curve,
    color='#4C72B0',
    linewidth=2
)

axes[1].axvline(
    MDE_RELATIVE * 100,
    color='#D62728',
    linestyle='--',
    linewidth=2,
    label=f'Configured MDE = {MDE_RELATIVE * 100:.0f}%'
)

axes[1].set_yscale('log')
axes[1].set_title("Sample Size vs MDE")
axes[1].set_xlabel("MDE (%)")
axes[1].set_ylabel("Users per Group (log)")
axes[1].legend()
axes[1].grid(alpha=0.3)

# --- Plot 3: Summary Score Bar Configurations ---
# TODO: Draw a bar layout chart explicitly mapping control vs treatment ultimate mean values on axes[2]
bar_labels = ['Control\nLightFM-only', 'Treatment\nHybrid']
bar_values = [ctrl_mean, treat_mean]
bar_colors = ['#4C72B0', '#55A868']

bars = axes[2].bar(
    bar_labels,
    bar_values,
    color=bar_colors,
    edgecolor='white'
)

max_bar_value = max(bar_values) if max(bar_values) > 0 else 1.0

for bar, value in zip(bars, bar_values):
    axes[2].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max_bar_value * 0.03,
        f"{value:.4f}",
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

axes[2].set_title("Offline NDCG@10 Comparison")
axes[2].set_ylabel("Mean NDCG@10")
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("output/04_ab_protocol.png", dpi=150, bbox_inches='tight')
plt.show()

print("    Saved -> output/04_ab_protocol.png")


# ---------------------------------------------------------------------------
# SECTION 8: Save A/B Protocol Results
# ---------------------------------------------------------------------------
print("\n[7] Saving A/B protocol artifacts...")

ab_results = {
    'control_ndcgs': control_ndcgs,
    'treatment_ndcgs': treatment_ndcgs,
    'control_evaluated_users': control_evaluated_users,
    'treatment_evaluated_users': treatment_evaluated_users,
    'control_mean_ndcg_at_10': ctrl_mean,
    'treatment_mean_ndcg_at_10': treat_mean,
    'absolute_difference': diff,
    'relative_lift_percent': rel_lift,
    't_statistic': t_stat,
    'one_tailed_p_value': p_one,
    'statistically_significant': statistically_significant,
    'experiment_config': {
        'k': K,
        'max_test_users': MAX_TEST_USERS,
        'alpha': alpha,
        'power': power,
        'baseline_ndcg': baseline_ndcg,
        'mde_relative': MDE_RELATIVE,
        'mde_absolute': MDE_ABSOLUTE,
        'required_per_group': n_per_group,
        'total_required': total_required,
        'random_state': RANDOM_STATE,
        'assignment_salt': 'ab_v1'
    }
}

artifact_path = "data/ab_protocol_artifacts.pkl"
temp_artifact_path = "data/ab_protocol_artifacts_temp.pkl"

with open(temp_artifact_path, "wb") as file:
    pickle.dump(
        ab_results,
        file,
        protocol=pickle.HIGHEST_PROTOCOL
    )

os.replace(temp_artifact_path, artifact_path)

print("    Saved -> data/ab_protocol_artifacts.pkl")


print("\n" + "=" * 60)
print("  LAB 3.4 COMPLETE — A/B PROTOCOL SUMMARY")
print("=" * 60)

print(f"""
  HYPOTHESIS:
  H0: Hybrid routing does not improve mean NDCG@10.
  H1: Hybrid routing improves mean NDCG@10.

  OFFLINE RESULTS:
  Control mean NDCG@10   : {ctrl_mean:.4f}
  Treatment mean NDCG@10 : {treat_mean:.4f}
  Relative lift          : {rel_lift:+.2f}%
  One-tailed p-value     : {p_one:.4f}

  POWER PLAN:
  Required users/group   : {n_per_group:,}
  Required total users   : {total_required:,}
  Target MDE             : {MDE_RELATIVE * 100:.1f}%

  OUTPUTS:
  [OK] output/04_ab_protocol.png
  [OK] data/ab_protocol_artifacts.pkl
""")
