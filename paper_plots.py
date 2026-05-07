"""Generate publication-quality figures from results/ CSV files."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.2)
os.makedirs("figures", exist_ok=True)

COLORS = {
    "LRU": "#2196F3", "LFU": "#4CAF50", "S3FIFO": "#FF9800",
    "Decay (best)": "#9C27B0", "Decay": "#9C27B0",
    "ML Regressor": "#F44336", "ML Ranker": "#E91E63", "ML": "#F44336",
    "Hybrid p=10%": "#00BCD4", "Hybrid Reg p=10%": "#00BCD4",
    "OPT (oracle)": "#607D8B", "OPT": "#607D8B",
}


def fig1_policy_comparison():
    """Bar charts: hit rate, mean response time, p99, WA across policies."""
    df = pd.read_csv("results/exp1_policy_comparison.csv")
    # S3FIFO now included (fixed in Change 17)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, col, label in [
        (axes[0, 0], "hit_rate", "Hit Rate (by bytes)"),
        (axes[0, 1], "wa", "Write Amplification"),
        (axes[1, 0], "mean_resp_ms", "Mean Response Time (ms)"),
        (axes[1, 1], "p99_resp_ms", "P99 Response Time (ms)"),
    ]:
        colors = [COLORS.get(p, "#999") for p in df["policy"]]
        bars = ax.bar(range(len(df)), df[col], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["policy"], rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(label)
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.3f}" if val < 10 else f"{val:.1f}",
                    ha="center", va="bottom", fontsize=8)

    fig.suptitle("Policy Comparison (10 GB Fast Tier, Shards 2-3)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig1_policy_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  figures/fig1_policy_comparison.png")


def fig2_latency_breakdown():
    """Grouped bars: hit vs miss latency, fast vs slow tier latency."""
    df = pd.read_csv("results/exp1_policy_comparison.csv")
    # S3FIFO now included (fixed in Change 17)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = range(len(df))
    w = 0.35

    ax = axes[0]
    ax.bar([i - w/2 for i in x], df["mean_hit_ms"], w, label="Hit (mean)", color="#4CAF50", edgecolor="black", linewidth=0.5)
    ax.bar([i + w/2 for i in x], df["mean_miss_ms"], w, label="Miss (mean)", color="#F44336", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["policy"], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Response Time (ms)")
    ax.set_title("Hit vs Miss Response Time")
    ax.legend()

    ax = axes[1]
    ax.bar([i - w/2 for i in x], df["p99_hit_ms"], w, label="Hit (p99)", color="#4CAF50", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.bar([i + w/2 for i in x], df["p99_miss_ms"], w, label="Miss (p99)", color="#F44336", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["policy"], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Response Time (ms)")
    ax.set_title("Hit vs Miss Tail Latency (P99)")
    ax.legend()

    plt.tight_layout()
    plt.savefig("figures/fig2_latency_breakdown.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  figures/fig2_latency_breakdown.png")


def fig3_tier_scaling():
    """Line plot: hit rate and response time vs fast tier size."""
    df = pd.read_csv("results/exp2_tier_scaling.csv")
    # S3FIFO now included (fixed in Change 17)

    # Append 100 GB results if available
    if os.path.exists("results/exp_100g.csv"):
        df100 = pd.read_csv("results/exp_100g.csv")
        policy_map = {
            "LRU": "LRU", "Decay (a=0.5 d=0.05)": "Decay",
            "ML Ranker": "ML Ranker", "Hybrid Reg p=10%": "Hybrid p=10%",
        }
        for _, row in df100.iterrows():
            mapped = policy_map.get(row["policy"])
            if mapped and mapped in df["policy"].values:
                df = pd.concat([df, pd.DataFrame([{
                    "fast_cap_mb": 102400, "policy": mapped,
                    "hit_rate": row["hit_rate"], "wa": row["wa"],
                    "mean_resp_ms": row["mean_resp_ms"], "p99_resp_ms": row["p99_resp_ms"],
                }])], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for policy in ["LRU", "Decay", "ML Ranker", "Hybrid p=10%"]:
        sub = df[df["policy"] == policy].sort_values("fast_cap_mb")
        if sub.empty:
            continue
        c = COLORS.get(policy, "#999")
        axes[0].plot(sub["fast_cap_mb"], sub["hit_rate"], "o-", color=c, label=policy, linewidth=2)
        axes[1].plot(sub["fast_cap_mb"], sub["mean_resp_ms"], "o-", color=c, label=policy, linewidth=2)

    for ax in axes:
        ax.set_xlabel("Fast Tier Capacity (MB)")
        ax.set_xscale("log", base=2)
        all_caps = sorted(df["fast_cap_mb"].unique())
        ax.set_xticks(all_caps)
        labels = []
        for c in all_caps:
            if c >= 1024:
                labels.append(f"{c//1024}G")
            else:
                labels.append(f"{c}M")
        ax.set_xticklabels(labels)
        ax.legend()

    axes[0].set_ylabel("Hit Rate")
    axes[0].set_title("Hit Rate vs Fast Tier Size")
    axes[1].set_ylabel("Mean Response Time (ms)")
    axes[1].set_title("Response Time vs Fast Tier Size")

    plt.tight_layout()
    plt.savefig("figures/fig3_tier_scaling.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  figures/fig3_tier_scaling.png")


def fig4_shard_distance():
    """Line plot: hit rate vs shard distance for LRU, ML, Decay."""
    df = pd.read_csv("results/exp3_shard_distance.csv")
    fig, ax = plt.subplots(figsize=(8, 5))

    for policy in df["policy"].unique():
        sub = df[df["policy"] == policy]
        c = COLORS.get(policy, "#999")
        ax.plot(sub["distance"], sub["hit_rate"], "o-", color=c, label=policy, linewidth=2, markersize=6)

    ax.set_xlabel("Shard Distance from Training Data")
    ax.set_ylabel("Hit Rate")
    ax.set_title("ML Generalization: Hit Rate vs Shard Distance")
    ax.legend()
    ax.axhline(y=df[df["policy"] == "LRU"]["hit_rate"].mean(), color=COLORS["LRU"],
               linestyle="--", alpha=0.3, label="_nolegend_")
    plt.tight_layout()
    plt.savefig("figures/fig4_shard_distance.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  figures/fig4_shard_distance.png")


def fig5_temporal():
    """Line plot: hit rate and response time over simulation time."""
    df = pd.read_csv("results/exp4_temporal.csv")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for policy in df["policy"].unique():
        sub = df[df["policy"] == policy]
        c = COLORS.get(policy, "#999")
        axes[0].plot(sub["time_s"], sub["hit_rate"], color=c, label=policy, alpha=0.8, linewidth=1.2)
        axes[1].plot(sub["time_s"], sub["mean_resp_ms"], color=c, label=policy, alpha=0.8, linewidth=1.2)

    axes[0].set_ylabel("Hit Rate (10s buckets)")
    axes[0].set_title("Hit Rate Over Simulation Time")
    axes[0].legend(loc="lower left")
    axes[1].set_xlabel("Simulation Time (s)")
    axes[1].set_ylabel("Mean Response Time (ms)")
    axes[1].set_title("Response Time Over Simulation Time")
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("figures/fig5_temporal.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  figures/fig5_temporal.png")


def fig6_training_size():
    """Bar/line plot: ML hit rate vs training data size, with LRU baseline."""
    df = pd.read_csv("results/exp5_training_size.csv")
    fig, ax = plt.subplots(figsize=(8, 5))

    for policy in df["policy"].unique():
        sub = df[df["policy"] == policy]
        c = COLORS.get(policy, "#999")
        style = "--" if policy == "LRU" else "o-"
        ax.plot(sub["train_rows"], sub["hit_rate"], style, color=c, label=policy, linewidth=2, markersize=6)

    ax.set_xlabel("Training Rows")
    ax.set_ylabel("Hit Rate")
    ax.set_title("Effect of Training Data Size on ML Eviction")
    ax.set_xscale("log")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/fig6_training_size.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  figures/fig6_training_size.png")


def fig7_ml_comparison():
    """Bar chart comparing all ML eviction approaches."""
    df = pd.read_csv("results/exp_ml_comparison.csv")
    # S3FIFO now included (fixed in Change 17)

    ML_COLORS = {
        "LRU (baseline)": "#2196F3", "OPT (oracle)": "#607D8B",
        "ML Regressor": "#F44336", "ML Ranker (LambdaRank)": "#E91E63",
        "Imitation Ranker": "#FF5722", "Hybrid Reg p=10%": "#00BCD4",
        "Hybrid Rank p=10%": "#009688", "Decay (no ML)": "#9C27B0",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, col, label in [
        (axes[0], "hit_rate", "Hit Rate"),
        (axes[1], "mean_resp_ms", "Mean Response Time (ms)"),
    ]:
        colors = [ML_COLORS.get(p, "#999") for p in df["policy"]]
        bars = ax.bar(range(len(df)), df[col], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["policy"], rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.3f}" if val < 10 else f"{val:.1f}",
                    ha="center", va="bottom", fontsize=7)

    fig.suptitle("ML Eviction Algorithm Comparison (10 GB Fast Tier, Cross-Shard)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig7_ml_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  figures/fig7_ml_comparison.png")


def fig8_placement_promotion():
    """Bar chart: placement/promotion strategies vs LRU."""
    df = pd.read_csv("results/exp_placement_promotion.csv")

    PP_COLORS = {
        "LRU (baseline)": "#2196F3", "Decay": "#9C27B0",
        "LRU+CA ratio<200": "#4CAF50", "LRU+CA ratio<100": "#66BB6A",
        "LRU+CA ratio<50": "#81C784", "LRU+CA ratio<10": "#A5D6A7",
        "Admission n=1": "#FF9800", "Size*Coldness evict": "#795548",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, col, label in [
        (axes[0], "hit_rate", "Hit Rate"),
        (axes[1], "mean_resp_ms", "Mean Response Time (ms)"),
        (axes[2], "wa", "Write Amplification"),
    ]:
        colors = [PP_COLORS.get(p, "#999") for p in df["policy"]]
        bars = ax.bar(range(len(df)), df[col], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["policy"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)
        for bar, val in zip(bars, df[col]):
            fmt = f"{val:.3f}" if val < 10 else f"{val:.1f}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    fmt, ha="center", va="bottom", fontsize=7)

    # Add LRU reference line on hit rate
    lru_hr = df[df["policy"] == "LRU (baseline)"]["hit_rate"].values[0]
    axes[0].axhline(y=lru_hr, color="#2196F3", linestyle="--", alpha=0.4)

    fig.suptitle("Placement & Promotion Strategies (10 GB Fast Tier)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig8_placement_promotion.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  figures/fig8_placement_promotion.png")


if __name__ == "__main__":
    print("Generating figures from results/ CSVs...")
    for name, fn in [
        ("fig1", fig1_policy_comparison),
        ("fig2", fig2_latency_breakdown),
        ("fig3", fig3_tier_scaling),
        ("fig4", fig4_shard_distance),
        ("fig5", fig5_temporal),
        ("fig6", fig6_training_size),
        ("fig7", fig7_ml_comparison),
        ("fig8", fig8_placement_promotion),
    ]:
        exp_map = {"fig1": "exp1_*", "fig2": "exp1_*", "fig3": "exp2_*",
                   "fig4": "exp3_*", "fig5": "exp4_*", "fig6": "exp5_*",
                   "fig7": "exp_ml_*", "fig8": "exp_placement_*"}
        import glob
        if not glob.glob(f"results/{exp_map[name]}.csv"):
            print(f"  SKIP {name}: no data file")
            continue
        try:
            fn()
        except Exception as e:
            print(f"  FAIL {name}: {e}")

    print("\nAll figures in figures/")
