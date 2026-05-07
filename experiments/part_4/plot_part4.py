"""Generate Part 4 plots from merged results."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PART_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(PART_DIR, "results")
FIG_DIR = os.path.join(PART_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

rows = []
with open(os.path.join(RESULT_DIR, "part4_all.csv")) as f:
    for r in csv.DictReader(f):
        for k in ["hit_rate", "wa", "mean_resp_ms", "p99_resp_ms"]:
            r[k] = float(r[k])
        rows.append(r)

# Fig 1: Cross-trained distance test with baseline reference
scenarios = ["baseline", "diff_group", "diff_cluster"]
sc_labels = ["Baseline\n(c1g1)", "Diff Group\n(c1g2)", "Diff Cluster\n(c3g1)"]
policies = ["LRU", "Decay", "ML_Reg", "OPT_Imit_Rank",
            "Hybrid_ML_Reg_p25", "Hybrid_OPT_Rank_p10"]
colors = {"LRU": "#333333", "Decay": "#666666",
          "ML_Reg": "#1f77b4", "OPT_Imit_Rank": "#d62728",
          "Hybrid_ML_Reg_p25": "#9467bd", "Hybrid_OPT_Rank_p10": "#e377c2"}
display = {"LRU": "LRU", "Decay": "Decay", "ML_Reg": "ML Reg",
           "OPT_Imit_Rank": "OPT Imit Rank",
           "Hybrid_ML_Reg_p25": "Hybrid ML Reg p25",
           "Hybrid_OPT_Rank_p10": "Hybrid OPT Rank p10"}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax_idx, (metric, title) in enumerate([("hit_rate", "Hit Rate"),
                                            ("mean_resp_ms", "Mean Resp (ms)")]):
    ax = axes[ax_idx]
    x = np.arange(len(scenarios))
    width = 0.12
    for i, pol in enumerate(policies):
        vals = []
        for sc in scenarios:
            match = [r for r in rows if r["policy"] == pol and r["scenario"] == sc]
            vals.append(match[0][metric] if match else 0)
        ax.bar(x + i * width, vals, width, label=display[pol], color=colors[pol])
    ax.set_xticks(x + width * (len(policies) - 1) / 2)
    ax.set_xticklabels(sc_labels, fontsize=9)
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("Part 4: Distance Testing — Cross-Trained Models (10GB)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "part4_distance_testing.png"), dpi=150)
print("Saved part4_distance_testing.png")

# Fig 2: Native vs cross-trained comparison
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
compare_scenarios = [
    ("diff_group", "native_c1g2", "c1g2"),
    ("diff_cluster", "native_c3g1", "c3g1"),
]
native_policies = ["LRU", "Decay", "ML_Reg", "ML_Rank", "OPT_Imit_Reg", "OPT_Imit_Rank"]
ncolors = {"LRU": "#333333", "Decay": "#666666", "ML_Reg": "#1f77b4",
           "ML_Rank": "#ff7f0e", "OPT_Imit_Reg": "#2ca02c", "OPT_Imit_Rank": "#d62728"}

for ax_idx, (cross_sc, native_sc, group_label) in enumerate(compare_scenarios):
    ax = axes2[ax_idx]
    x = np.arange(len(native_policies))
    width = 0.35

    cross_vals = []
    native_vals = []
    for pol in native_policies:
        cm = [r for r in rows if r["policy"] == pol and r["scenario"] == cross_sc]
        nm = [r for r in rows if r["policy"] == pol and r["scenario"] == native_sc]
        cross_vals.append(cm[0]["hit_rate"] if cm else 0)
        native_vals.append(nm[0]["hit_rate"] if nm else 0)

    ax.bar(x - width/2, cross_vals, width, label="Cross-trained (c1g1)", color="#ff9999")
    ax.bar(x + width/2, native_vals, width, label=f"Native ({group_label})", color="#99ccff")
    ax.set_xticks(x)
    ax.set_xticklabels(native_policies, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Hit Rate")
    ax.set_title(f"Eval on {group_label}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

fig2.suptitle("Part 4: Native vs Cross-Trained Models", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "part4_native_vs_cross.png"), dpi=150)
print("Saved part4_native_vs_cross.png")
