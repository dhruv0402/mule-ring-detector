"""Generate publication-ready figures for Capstone Report and Presentation Slides."""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score

sns.set_theme(style="whitegrid")
plt.rcParams["font.size"] = 11

out_dir = Path("reports/figures")
out_dir.mkdir(parents=True, exist_ok=True)

print("Loading data for figure generation...")
preds = pd.read_parquet("reports/test_predictions.parquet")
y_true = preds["label"].to_numpy()
s1 = preds["score_m1"].to_numpy()
s2 = preds["score_m2"].to_numpy()

# ----------------- FIGURE 1: Precision-Recall Curves -----------------
print("Generating Figure 1: Precision-Recall Curves...")
p1, r1, _ = precision_recall_curve(y_true, s1)
p2, r2, _ = precision_recall_curve(y_true, s2)
ap1 = average_precision_score(y_true, s1)
ap2 = average_precision_score(y_true, s2)

plt.figure(figsize=(9, 6))
plt.plot(r1, p1, label=f"M1: Raw Tabular Baseline (PR-AUC = {ap1:.4f})", color="#7f8c8d", lw=2, ls="--")
plt.plot(r2, p2, label=f"M2: Causal Graph-Augmented (PR-AUC = {ap2:.4f})", color="#2980b9", lw=2.5)
plt.axhline(y=y_true.mean(), color="#e74c3c", ls=":", label=f"Random Base Rate ({y_true.mean():.4%})")

plt.title("Precision-Recall Curve Comparison (IBM HI-Small Test Partition)", fontsize=13, fontweight="bold")
plt.xlabel("Recall (Minority Class)", fontsize=11)
plt.ylabel("Precision (Minority Class)", fontsize=11)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 0.4])
plt.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig(out_dir / "pr_curves.png", dpi=300)
plt.close()

# ----------------- FIGURE 2: Feature Importance -----------------
print("Generating Figure 2: Feature Importance...")
m2_path = Path("reports/model_m2.joblib")
if m2_path.exists():
    m2 = joblib.load(m2_path)
    feat_names = m2.feature_name_
    feat_imps = m2.feature_importances_
    fi_df = pd.DataFrame({"Feature": feat_names, "Importance": feat_imps})
    fi_df = fi_df.sort_values("Importance", ascending=False).head(12)

    plt.figure(figsize=(10, 6))
    colors = ["#2980b9" if any(k in f for k in ["distinct", "ratio", "reverse", "balance", "count", "mean"]) else "#95a5a6" for f in fi_df["Feature"]]
    sns.barplot(x="Importance", y="Feature", data=fi_df, hue="Feature", palette=colors, legend=False)
    plt.title("Top Feature Importances in Model M2 (LightGBM)", fontsize=13, fontweight="bold")
    plt.xlabel("Split Feature Importance", fontsize=11)
    plt.ylabel("Engineered Feature", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=300)
    plt.close()

# ----------------- FIGURE 3: Laundering Topology Recovery -----------------
print("Generating Figure 3: Topology Recovery Breakdown...")
patterns = ["RANDOM", "CYCLE", "SCATTER-GATHER", "GATHER-SCATTER", "FAN-OUT", "FAN-IN", "BIPARTITE", "STACK"]
recall_j50 = [64.7, 56.0, 47.8, 51.3, 33.3, 55.0, 9.5, 3.8]
recall_j30 = [82.4, 80.0, 60.9, 59.0, 57.1, 55.0, 9.5, 3.8]

x = np.arange(len(patterns))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, recall_j50, width, label="Strict Overlap (Jaccard >= 0.5)", color="#1f77b4")
plt.bar(x + width/2, recall_j30, width, label="Moderate Overlap (Jaccard >= 0.3)", color="#aec7e8")

plt.ylabel("Ring Recall Rate (%)", fontsize=11)
plt.title("Ring Recovery Rate Across 8 Laundering Topologies (192 Test Rings)", fontsize=13, fontweight="bold")
plt.xticks(x, patterns, rotation=25, ha="right")
plt.ylim([0, 100])
for i in range(len(patterns)):
    plt.text(x[i] - width/2, recall_j50[i] + 1.5, f"{recall_j50[i]:.1f}%", ha="center", fontsize=9)
    plt.text(x[i] + width/2, recall_j30[i] + 1.5, f"{recall_j30[i]:.1f}%", ha="center", fontsize=9)

plt.axhline(50, color="#7f8c8d", ls="--", alpha=0.5)
plt.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig(out_dir / "topology_recall.png", dpi=300)
plt.close()

# ----------------- FIGURE 4: Alert Budget Trade-off -----------------
print("Generating Figure 4: Alert Budget Trade-off Curve...")
ks = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
p_at_k = []
r_at_k = []
sorted_indices = np.argsort(-s2)

for k in ks:
    top_y = y_true[sorted_indices[:k]]
    prec = top_y.mean()
    rec = top_y.sum() / max(y_true.sum(), 1)
    p_at_k.append(prec * 100)
    r_at_k.append(rec * 100)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

line1 = ax1.plot(ks, p_at_k, color="#e74c3c", lw=2.5, marker="o", label="Precision @ K (%)")
line2 = ax2.plot(ks, r_at_k, color="#2980b9", lw=2.5, marker="s", label="Recall @ K (%)")

ax1.set_xlabel("Alert Budget (Top K Scored Transactions)", fontsize=11)
ax1.set_ylabel("Precision @ K (%)", color="#e74c3c", fontsize=11)
ax2.set_ylabel("Recall @ K (%)", color="#2980b9", fontsize=11)
ax1.set_xscale("log")
plt.title("Investigator Alert Budget Trade-Off: Precision vs. Recall at Top K", fontsize=13, fontweight="bold")

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="center right", frameon=True)
plt.tight_layout()
plt.savefig(out_dir / "alert_budget_curve.png", dpi=300)
plt.close()

print(f"All figures saved to {out_dir}/ successfully!")
