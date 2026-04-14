import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_loader import load_data
from src.preprocessing.features import add_water_loss_feats
from src.model.iso_forest_model import ScratchIsolationForest

df_train = load_data("data/BATADAL_dataset03.csv")
df_test = load_data("data/BATADAL_dataset04.csv")

df_train = add_water_loss_feats(df_train)
df_test = add_water_loss_feats(df_test)

cols = [c for c in df_train.columns if c != "ATT_FLAG"]
X_train = df_train[cols].values
X_test = df_test[cols].values

mn = np.median(X_train, axis=0)
q3 = np.percentile(X_train, 75, axis=0)
q1 = np.percentile(X_train, 25,  axis=0)
iqr = (q3-q1) + 1e-6

X_train_scaled = (X_train - mn) / iqr
X_test_scaled = (X_test - mn) / iqr

print("Model is being trained")
model = ScratchIsolationForest(n_trees=100)
model.fit(X_train_scaled)

print("Checking Test data for anomalies")
scores = model.compute_anomaly_score(X_test_scaled)

threshold = np.percentile(scores, 95)
df_test["anomaly_score"] = scores
df_test["is_anomaly"] = (scores > threshold).astype(int)

anomalies = df_test[df_test["is_anomaly"] == 1]

os.makedirs("outputs", exist_ok=True)

# =============================================================================
# PLOT 1 — Main detection view
# Two panels: flow signal on top, anomaly score on bottom
# Both share the same x-axis so events line up visually
# =============================================================================
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(16, 8),
    sharex=True,                      # same x-axis — events align vertically
    gridspec_kw={"height_ratios": [2, 1]}  # top panel twice as tall
)

fig.suptitle(
    "South African Water Guard — Unsupervised Leak Detection\n"
    "BATADAL Dataset · Pump 1 (F_PU1) · Test Period Jul–Dec 2016",
    fontsize=13, fontweight="bold", y=1.01
)

# ── Top panel: flow signal + anomaly markers ──────────────────────────────
ax1.plot(
    df_test.index, df_test["F_PU1"],
    color="steelblue", linewidth=0.7, alpha=0.8,
    label="Flow rate — Pump 1 (LPS)"
)

ax1.scatter(
    anomalies.index, anomalies["F_PU1"],
    color="red", s=18, zorder=5,
    label=f"AI detected anomaly ({len(anomalies)} flags)"
)

ax1.set_ylabel("Flow rate (litres per second)", fontsize=11)
ax1.legend(loc="upper right", fontsize=9)
ax1.set_title("Flow signal with anomaly flags", fontsize=10, pad=6)

# ── Bottom panel: raw anomaly score ──────────────────────────────────────
ax2.plot(
    df_test.index, scores,
    color="purple", linewidth=0.7, alpha=0.85,
    label="Anomaly score"
)

# Shade the area above the threshold in red so it's immediately obvious
ax2.fill_between(
    df_test.index, threshold, scores,
    where=(scores >= threshold),
    color="red", alpha=0.25,
    label="Above threshold (flagged)"
)

ax2.axhline(
    y=threshold, color="red", linestyle="--",
    linewidth=1.2, label=f"Threshold ({threshold:.3f})"
)

ax2.set_ylabel("Anomaly score\n(higher = more suspicious)", fontsize=11)
ax2.set_xlabel("Date", fontsize=11)
ax2.legend(loc="upper right", fontsize=9)
ax2.set_title("Raw Isolation Forest score over time", fontsize=10, pad=6)

plt.tight_layout()
plot1_path = "outputs/plot1_detection.png"
plt.savefig(plot1_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {plot1_path}")


# =============================================================================
# PLOT 2 — Score distribution histogram
# Shows how well the model separates normal from anomalous readings
# A good model = two distinct humps that don't overlap much
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5))

normal_scores  = scores[df_test["is_anomaly"].values == 0]
anomaly_scores = scores[df_test["is_anomaly"].values == 1]

ax.hist(
    normal_scores, bins=60,
    color="steelblue", alpha=0.6, density=True,
    label=f"Normal readings ({len(normal_scores)})"
)
ax.hist(
    anomaly_scores, bins=30,
    color="red", alpha=0.6, density=True,
    label=f"Flagged anomalies ({len(anomaly_scores)})"
)

ax.axvline(
    x=threshold, color="red", linestyle="--",
    linewidth=1.5, label=f"Decision threshold ({threshold:.3f})"
)

ax.set_xlabel("Anomaly score (higher = more anomalous)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title(
    "Anomaly Score Distribution\n"
    "Well-separated humps = model is confidently distinguishing normal vs anomalous",
    fontsize=11
)
ax.legend(fontsize=9)

plt.tight_layout()
plot2_path = "outputs/plot2_score_distribution.png"
plt.savefig(plot2_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {plot2_path}")


# =============================================================================
# PLOT 3 — Multi-sensor overview (small multiples)
# Shows the 6 pump flow sensors side by side
# Lets you see which sensors triggered anomalies and which look clean
# =============================================================================
pump_cols = [c for c in df_test.columns if c.startswith("F_PU") and "_" not in c[4:]]
pump_cols = pump_cols[:6]   # cap at 6 so the grid stays readable

n_cols = 2
n_rows = (len(pump_cols) + 1) // n_cols

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(16, n_rows * 3),
    sharex=True
)
axes = axes.flatten()   # makes it easier to loop — no nested indexing

fig.suptitle(
    "Multi-Sensor Overview — All Pump Flow Sensors\n"
    "Red = AI flagged anomaly",
    fontsize=13, fontweight="bold"
)

for i, col in enumerate(pump_cols):
    ax = axes[i]

    ax.plot(
        df_test.index, df_test[col],
        color="steelblue", linewidth=0.6, alpha=0.7
    )

    ax.scatter(
        anomalies.index, anomalies[col],
        color="red", s=8, zorder=5
    )

    # count how many anomalies hit this sensor's time window
    n_flags = len(anomalies)
    ax.set_title(f"{col}  ·  {n_flags} anomaly timestamps", fontsize=9)
    ax.set_ylabel("LPS", fontsize=8)
    ax.tick_params(labelsize=7)

# hide any unused subplots if pump_cols count is odd
for j in range(len(pump_cols), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plot3_path = "outputs/plot3_multi_sensor.png"
plt.savefig(plot3_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {plot3_path}")


# =============================================================================
# PLOT 4 — Monthly anomaly count bar chart
# Answers: "which month had the most anomalies?"
# Useful for reporting to a municipality or operations team
# =============================================================================
df_test["month"] = df_test.index.to_period("M").astype(str)
monthly_counts   = df_test.groupby("month")["is_anomaly"].sum()

fig, ax = plt.subplots(figsize=(9, 5))

bar_colors = [
    "#E24B4A" if v == monthly_counts.max()   # highlight worst month in red
    else "#BA7517" if v > monthly_counts.mean()
    else "#639922"
    for v in monthly_counts.values
]

bars = ax.bar(
    monthly_counts.index, monthly_counts.values,
    color=bar_colors, edgecolor="white", width=0.6
)

# add count label on top of each bar
for bar, val in zip(bars, monthly_counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        str(int(val)),
        ha="center", va="bottom", fontsize=10, fontweight="500"
    )

ax.set_xlabel("Month", fontsize=11)
ax.set_ylabel("Number of anomalous hours flagged", fontsize=11)
ax.set_title(
    "Monthly Anomaly Count\n"
    "Red = worst month  ·  Amber = above average  ·  Green = below average",
    fontsize=11
)

plt.tight_layout()
plot4_path = "outputs/plot4_monthly_anomalies.png"
plt.savefig(plot4_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {plot4_path}")


# =============================================================================
# PLOT 5 — Minimum night flow over time
# The water industry's core leak indicator
# A rising "floor" in this chart = background leak growing over time
# =============================================================================
mnf_col = "F_PU1_mnf"   # created by add_water_loss_feats

if mnf_col in df_test.columns:
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(
        df_test.index, df_test[mnf_col],
        color="green", linewidth=1.0, alpha=0.85,
        label="Min night flow — Pump 1 (rolling 24hr)"
    )
    ax.fill_between(
        df_test.index, 0, df_test[mnf_col],
        color="green", alpha=0.15
    )

    # mark anomaly periods on this chart too
    ax.scatter(
        anomalies.index, anomalies[mnf_col],
        color="red", s=12, zorder=5,
        label="Anomaly flag"
    )

    ax.set_ylabel("Min night flow (LPS)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_title(
        "Minimum Night Flow — Key Leak Indicator\n"
        "High values at 1–4AM when demand should be near zero = likely background leak",
        fontsize=11
    )
    ax.legend(fontsize=9)

    plt.tight_layout()
    plot5_path = "outputs/plot5_night_flow.png"
    plt.savefig(plot5_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot5_path}")

# ── Summary ────────────────────────────────────────────────────────────────
print()
print("=" * 50)
print("RESULTS SUMMARY")
print("=" * 50)
print(f"Test period:        {df_test.index.min().date()} → {df_test.index.max().date()}")
print(f"Total hours scored: {len(df_test)}")
print(f"Anomalies flagged:  {len(anomalies)} ({100*len(anomalies)/len(df_test):.1f}%)")
print(f"Threshold used:     {threshold:.4f} (95th percentile of scores)")
print(f"Worst month:        {monthly_counts.idxmax()} ({monthly_counts.max()} flags)")
print()
print("Output files:")
for p in [plot1_path, plot2_path, plot3_path, plot4_path, plot5_path]:
    print(f"  {p}")