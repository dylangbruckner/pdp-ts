"""Part 7: Promotion and Placement strategies.
Tests ML-guided promotion (gate which files get promoted to fast tier) and
cost-aware filtering (skip promoting huge files for tiny IOs).
All use LRU eviction. The hypothesis: ML can add value in promotion decisions
even though it fails at eviction, because promotion is a binary decision
(promote/skip) with a clear cost-benefit tradeoff.
"""
import csv, json, os, sys, time, warnings
warnings.filterwarnings("ignore")

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PART_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)

from src.config import EvaluatorConfig, TierConfig
from src.evaluator import Evaluator
from src.trace_loader import load_thesios_csv
from src.policy import make_lru_policy, make_ml_policy
from src.decay_policy import make_decay_policy
from src.cost_aware import wrap_cost_aware
from src.ml_promotion_policy import make_ml_promotion_policy

GB, MB = 1024**3, 1024**2
FAST_CAP = 10 * GB
TRACE_DIR = os.path.join(PROJECT, "traces/workload2/cluster_1_group_1")
SHARDS = sorted(os.listdir(TRACE_DIR))
RESULT_DIR = os.path.join(PART_DIR, "results")
FIG_DIR = os.path.join(PART_DIR, "figures")

SLOW_CFG = TierConfig(200*MB, 150*MB, 5e-3, 5e-3, 8, 8)
FAST_CFG = TierConfig(3.5*GB, 3.0*GB, 20e-6, 20e-6, 32, 32, FAST_CAP)

CSV_FIELDS = ["policy", "category", "hit_rate", "wa",
              "mean_resp_ms", "p99_resp_ms", "wall_s"]
RESULTS_CSV = os.path.join(RESULT_DIR, "part7_results.csv")
CHECKPOINT = os.path.join(RESULT_DIR, "checkpoint.json")

def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT) as f:
            return json.load(f)
    return {"completed": []}

def save_checkpoint(ck):
    with open(CHECKPOINT, "w") as f:
        json.dump(ck, f, indent=2)

def is_done(ck, key):
    return key in ck["completed"]

def mark_done(ck, key):
    ck["completed"].append(key)
    save_checkpoint(ck)

def append_result(row):
    exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)

def shard_path(idx):
    return os.path.join(TRACE_DIR, SHARDS[idx])

def load_eval_data():
    reqs = []
    for i in range(4, 6):
        reqs.extend(load_thesios_csv(shard_path(i)))
    reqs.sort(key=lambda r: r.arrival_time)
    warmup_n = len(reqs) // 2
    for r in reqs[:warmup_n]:
        r.is_warmup = True
    return reqs, warmup_n

def run_eval(name, pol, reqs, warmup_n, category=""):
    cfg = EvaluatorConfig(
        FAST_CFG, SLOW_CFG, shard_path(0), "/dev/null",
        warmup_ops=warmup_n, always_write_slow=True, on_fast_full="spill",
    )
    t0 = time.perf_counter()
    mc = Evaluator(cfg, pol, reqs).run()
    elapsed = time.perf_counter() - t0
    s = mc.global_metrics.summary()
    wa = mc.global_metrics.write_amplification(mc._requested_write_bytes)
    row = {
        "policy": name, "category": category,
        "hit_rate": round(s["hit_rate"], 6), "wa": round(wa, 4),
        "mean_resp_ms": round(s["mean_response_time_s"] * 1000, 4),
        "p99_resp_ms": round(s["p99_response_time_s"] * 1000, 4),
        "wall_s": round(elapsed, 1),
    }
    print(f"  {name:<40s} hit={row['hit_rate']:.4f}  WA={row['wa']:.3f}  "
          f"mean={row['mean_resp_ms']:.3f}ms  ({elapsed:.0f}s)")
    return row


if __name__ == "__main__":
    ck = load_checkpoint()
    reqs, warmup_n = load_eval_data()
    print(f"Eval: {len(reqs)} requests, {warmup_n} warmup\n")

    # Baselines
    print("=== Baselines ===")
    baselines = [
        ("LRU", "baseline", lambda: make_lru_policy(FAST_CAP)),
        ("Decay", "baseline", lambda: make_decay_policy(FAST_CAP, alpha=0.5,
                                    decay_rate=0.05, fast_fill_threshold=0.9)),
    ]
    for name, cat, make_fn in baselines:
        if is_done(ck, name):
            print(f"  {name} done.")
            continue
        row = run_eval(name, make_fn(), reqs, warmup_n, cat)
        append_result(row)
        mark_done(ck, name)

    # Cost-aware promotion filtering (heuristic, no ML)
    print("\n=== Cost-Aware Promotion (LRU eviction + promotion filter) ===")
    ca_configs = [
        ("LRU+CA_r50_n1", 50.0, 1),
        ("LRU+CA_r100_n1", 100.0, 1),
        ("LRU+CA_r200_n1", 200.0, 1),
        ("LRU+CA_r50_n2", 50.0, 2),
        ("LRU+CA_r100_n2", 100.0, 2),
        ("LRU+CA_r200_n2", 200.0, 2),
        ("LRU+CA_r50_n3", 50.0, 3),
    ]
    for name, ratio, min_acc in ca_configs:
        if is_done(ck, name):
            print(f"  {name} done.")
            continue
        base = make_lru_policy(FAST_CAP)
        pol = wrap_cost_aware(base, max_file_io_ratio=ratio,
                              min_accesses_to_promote=min_acc)
        row = run_eval(name, pol, reqs, warmup_n, "cost_aware")
        append_result(row)
        mark_done(ck, name)

    # Cost-aware on Decay
    print("\n=== Cost-Aware on Decay ===")
    for name, ratio, min_acc in [("Decay+CA_r100_n2", 100.0, 2), ("Decay+CA_r200_n1", 200.0, 1)]:
        if is_done(ck, name):
            print(f"  {name} done.")
            continue
        base = make_decay_policy(FAST_CAP, alpha=0.5, decay_rate=0.05, fast_fill_threshold=0.9)
        pol = wrap_cost_aware(base, max_file_io_ratio=ratio, min_accesses_to_promote=min_acc)
        row = run_eval(name, pol, reqs, warmup_n, "cost_aware_decay")
        append_result(row)
        mark_done(ck, name)

    # ML-guided promotion (LRU eviction, ML gates promotions)
    print("\n=== ML Promotion (LRU eviction + ML promotion gate) ===")
    ml_promo_configs = [
        ("ML_Promo_hr_t0", "hit_rate", 0.0),
        ("ML_Promo_hr_t0.5", "hit_rate", 0.5),
        ("ML_Promo_hr_t1", "hit_rate", 1.0),
        ("ML_Promo_hr_t2", "hit_rate", 2.0),
        ("ML_Promo_hr_t5", "hit_rate", 5.0),
        ("ML_Promo_rt_t0", "response_time", 0.0),
        ("ML_Promo_rt_t0.5", "response_time", 0.5),
        ("ML_Promo_rt_t1", "response_time", 1.0),
        ("ML_Promo_rt_t2", "response_time", 2.0),
    ]
    for name, obj, thresh in ml_promo_configs:
        if is_done(ck, name):
            print(f"  {name} done.")
            continue
        pol = make_ml_promotion_policy(FAST_CAP, objective=obj,
                                        promote_threshold=thresh,
                                        fast_fill_threshold=0.9)
        row = run_eval(name, pol, reqs, warmup_n, "ml_promotion")
        append_result(row)
        mark_done(ck, name)

    # Combined: ML promotion + cost-aware
    print("\n=== ML Promotion + Cost-Aware ===")
    combos = [
        ("ML_Promo_hr_t1+CA_r100", "hit_rate", 1.0, 100.0, 2),
        ("ML_Promo_hr_t2+CA_r100", "hit_rate", 2.0, 100.0, 2),
        ("ML_Promo_rt_t1+CA_r100", "response_time", 1.0, 100.0, 2),
    ]
    for name, obj, thresh, ratio, min_acc in combos:
        if is_done(ck, name):
            print(f"  {name} done.")
            continue
        base = make_ml_promotion_policy(FAST_CAP, objective=obj,
                                         promote_threshold=thresh,
                                         fast_fill_threshold=0.9)
        pol = wrap_cost_aware(base, max_file_io_ratio=ratio,
                              min_accesses_to_promote=min_acc)
        row = run_eval(name, pol, reqs, warmup_n, "ml_promo_ca")
        append_result(row)
        mark_done(ck, name)

    # Plot
    print("\n=== Generating Plots ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = []
    with open(RESULTS_CSV) as f:
        for r in csv.DictReader(f):
            if r["category"] == "ml_promo_ca":
                continue
            for k in ["hit_rate", "wa", "mean_resp_ms", "p99_resp_ms"]:
                r[k] = float(r[k])
            rows.append(r)

    # Fig 1: All policies comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    metrics = [("hit_rate", "Hit Rate"), ("wa", "Write Amplification"),
               ("mean_resp_ms", "Mean Response Time (ms)"),
               ("p99_resp_ms", "P99 Response Time (ms)")]

    cat_colors = {"baseline": "#333333", "cost_aware": "#2ca02c",
                  "cost_aware_decay": "#8fbc8f",
                  "ml_promotion": "#d62728", "ml_promo_ca": "#9467bd"}

    for ax_idx, (metric, title) in enumerate(metrics):
        ax = axes.flatten()[ax_idx]
        names = [r["policy"] for r in rows]
        vals = [r[metric] for r in rows]
        colors = [cat_colors.get(r["category"], "gray") for r in rows]
        bars = ax.barh(range(len(names)), vals, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xlabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#333333", label="Baseline"),
        Patch(facecolor="#2ca02c", label="Cost-Aware (LRU)"),
        Patch(facecolor="#8fbc8f", label="Cost-Aware (Decay)"),
        Patch(facecolor="#d62728", label="ML Promotion"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9)

    fig.suptitle("Part 7: Promotion & Placement Strategies (10GB)", fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(os.path.join(FIG_DIR, "part7_promotion_placement.png"), dpi=150)
    print(f"Saved part7_promotion_placement.png")

    # Fig 2: Hit rate vs WA tradeoff
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    for r in rows:
        c = cat_colors.get(r["category"], "gray")
        ax2.scatter(r["wa"], r["hit_rate"], c=c, s=80, zorder=5)
        ax2.annotate(r["policy"], (r["wa"], r["hit_rate"]),
                    fontsize=5, ha="left", va="bottom")
    ax2.set_xlabel("Write Amplification")
    ax2.set_ylabel("Hit Rate")
    ax2.set_title("Part 7: Hit Rate vs Write Amplification Tradeoff")
    ax2.legend(handles=legend_elements, fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "part7_tradeoff.png"), dpi=150)
    print(f"Saved part7_tradeoff.png")

    print("\nDone!")
