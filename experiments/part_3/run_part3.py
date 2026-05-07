"""Part 3: Eviction policy comparison at 10GB.
Sub-experiment A: All policies (LRU, LFU, Decay, OPT + 4 ML) with optimal configs.
Sub-experiment B: LRU+ML Hybrid at 0%, 10%, 25%, 50%, 100% for each ML model.
Saturation: shards 4-7, Evaluation: shards 8-11.
"""
import csv, json, os, sys, time, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PART_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)

from src.config import EvaluatorConfig, TierConfig
from src.evaluator import Evaluator
from src.trace_loader import load_thesios_csv
from src.policy import make_lru_policy, make_ml_policy, make_opt_policy, make_hybrid_policy
from src.decay_policy import make_decay_policy

GB, MB = 1024**3, 1024**2
FAST_CAP = 10 * GB
TRACE_DIR = os.path.join(PROJECT, "traces/workload2/cluster_1_group_1")
SHARDS = sorted(os.listdir(TRACE_DIR))
MODEL_DIR = os.path.join(PART_DIR, "models")
RESULT_DIR = os.path.join(PART_DIR, "results")
FIG_DIR = os.path.join(PART_DIR, "figures")
for d in [MODEL_DIR, RESULT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

SLOW_CFG = TierConfig(200*MB, 150*MB, 5e-3, 5e-3, 8, 8)
FAST_CFG = TierConfig(3.5*GB, 3.0*GB, 20e-6, 20e-6, 32, 32, FAST_CAP)

CSV_FIELDS = ["policy", "sub_experiment", "hybrid_pct", "hit_rate", "wa",
              "mean_resp_ms", "p99_resp_ms", "wall_s"]
RESULTS_CSV = os.path.join(RESULT_DIR, "part3_results.csv")
_BATCH_ID = None

def _ck_path():
    if _BATCH_ID is not None:
        return os.path.join(RESULT_DIR, f"checkpoint_batch{_BATCH_ID}.json")
    return os.path.join(RESULT_DIR, "checkpoint.json")

def _csv_path():
    if _BATCH_ID is not None:
        return os.path.join(RESULT_DIR, f"part3_batch{_BATCH_ID}.csv")
    return RESULTS_CSV

def load_checkpoint():
    p = _ck_path()
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"completed": []}

def save_checkpoint(ck):
    with open(_ck_path(), "w") as f:
        json.dump(ck, f, indent=2)

def is_done(ck, key):
    return key in ck["completed"]

def mark_done(ck, key):
    ck["completed"].append(key)
    save_checkpoint(ck)

def append_result(row):
    path = _csv_path()
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)

def shard_path(idx):
    return os.path.join(TRACE_DIR, SHARDS[idx])

def load_shard_range(start, end):
    reqs = []
    for i in range(start, end):
        reqs.extend(load_thesios_csv(shard_path(i)))
    reqs.sort(key=lambda r: r.arrival_time)
    return reqs

def load_eval_data():
    """Shards 4-5 with 50% same-population warmup (consistent with Parts 1-2)."""
    reqs = load_shard_range(4, 6)
    warmup_n = len(reqs) // 2
    for r in reqs[:warmup_n]:
        r.is_warmup = True
    return reqs, warmup_n

def run_eval(name, pol, reqs, warmup_n, sub_exp="", hybrid_pct=""):
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
        "policy": name, "sub_experiment": sub_exp, "hybrid_pct": hybrid_pct,
        "hit_rate": round(s["hit_rate"], 6), "wa": round(wa, 4),
        "mean_resp_ms": round(s["mean_response_time_s"] * 1000, 4),
        "p99_resp_ms": round(s["p99_response_time_s"] * 1000, 4),
        "wall_s": round(elapsed, 1),
    }
    print(f"  {name:<45s} hit={row['hit_rate']:.4f}  mean={row['mean_resp_ms']:.3f}ms  ({elapsed:.0f}s)")
    return row


def get_model_path(model_type):
    """Get the best model path for each type based on Part 1/2 results."""
    part1_models = os.path.join(PROJECT, "experiments/part_1/models")
    part2_models = os.path.join(PROJECT, "experiments/part_2/models")
    paths = {
        "ML_Reg": os.path.join(part2_models, "ML_Reg_t1000_d10_l127.joblib"),
        "ML_Rank": os.path.join(part2_models, "ML_Rank_t500_d8_l63.joblib"),
        "OPT_Imit_Reg": os.path.join(part2_models, "OPT_Reg_t500_d6_l31.joblib"),
        "OPT_Imit_Rank": os.path.join(part2_models, "OPT_Rank_t100_d4_l15.joblib"),
    }
    p = paths.get(model_type)
    if p and not os.path.exists(p):
        fallbacks = {
            "ML_Reg": os.path.join(part1_models, "ml_reg_4shards.joblib"),
            "ML_Rank": os.path.join(part1_models, "ml_rank_4shards.joblib"),
            "OPT_Imit_Reg": os.path.join(part1_models, "opt_reg_1shard.joblib"),
            "OPT_Imit_Rank": os.path.join(part1_models, "opt_rank_10k.joblib"),
        }
        p = fallbacks.get(model_type)
    return p


def run_basic_comparison(ck, reqs, warmup_n, ml_only=False):
    """Sub-experiment A: all policies."""
    from src.policy import make_lfu_policy

    if not ml_only:
        heuristics = [
            ("LRU", lambda: make_lru_policy(FAST_CAP)),
            ("LFU", lambda: make_lfu_policy(FAST_CAP)),
            ("Decay", lambda: make_decay_policy(FAST_CAP, alpha=0.5, decay_rate=0.05,
                                                 fast_fill_threshold=0.9)),
            ("OPT", lambda: make_opt_policy(FAST_CAP, reqs, fast_fill_threshold=0.9,
                                             background_interval=5.0)),
        ]
        for name, make_fn in heuristics:
            key = f"basic_{name}"
            if is_done(ck, key):
                print(f"  {name} already done.")
                continue
            pol = make_fn()
            row = run_eval(name, pol, reqs, warmup_n, sub_exp="basic")
            append_result(row)
            mark_done(ck, key)

    for mt in ["ML_Reg", "ML_Rank", "OPT_Imit_Reg", "OPT_Imit_Rank"]:
        key = f"basic_{mt}"
        if is_done(ck, key):
            print(f"  {mt} already done.")
            continue
        model_path = get_model_path(mt)
        if not model_path or not os.path.exists(model_path):
            print(f"  {mt}: model not found, skipping")
            continue
        pol = make_ml_policy(FAST_CAP, model_path, fast_fill_threshold=0.9)
        row = run_eval(mt, pol, reqs, warmup_n, sub_exp="basic")
        append_result(row)
        mark_done(ck, key)


def run_hybrid_sweep(ck, reqs, warmup_n):
    """Sub-experiment B: hybrid LRU+ML at various protection percentages."""
    hybrid_pcts = [0.10, 0.25, 0.50]

    for mt in ["ML_Reg", "ML_Rank", "OPT_Imit_Reg", "OPT_Imit_Rank"]:
        model_path = get_model_path(mt)
        if not model_path or not os.path.exists(model_path):
            print(f"  {mt}: model not found, skipping hybrid sweep")
            continue

        for pct in hybrid_pcts:
            name = f"Hybrid_{mt}_p{int(pct*100)}"
            key = f"hybrid_{name}"
            if is_done(ck, key):
                print(f"  {name} already done.")
                continue
            pol = make_hybrid_policy(FAST_CAP, model_path,
                                     protect_percentile=pct,
                                     fast_fill_threshold=0.9)
            row = run_eval(name, pol, reqs, warmup_n,
                          sub_exp="hybrid", hybrid_pct=str(int(pct*100)))
            append_result(row)
            mark_done(ck, key)


def make_plots():
    import glob as _glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_rows = []
    seen = set()
    for f in sorted(_glob.glob(os.path.join(RESULT_DIR, "part3_batch*.csv"))):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r["policy"] not in seen:
                    all_rows.append(r)
                    seen.add(r["policy"])
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV) as fh:
            for r in csv.DictReader(fh):
                if r["policy"] not in seen:
                    all_rows.append(r)
                    seen.add(r["policy"])
    with open(RESULTS_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Merged {len(all_rows)} results")

    for r in all_rows:
        for k in ["hit_rate", "wa", "mean_resp_ms", "p99_resp_ms"]:
            r[k] = float(r[k])

    # Fig A: basic comparison bar chart
    basic = [r for r in all_rows if r["sub_experiment"] == "basic"]
    if basic:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = [("hit_rate", "Hit Rate"), ("wa", "Write Amplification"),
                   ("mean_resp_ms", "Mean Response Time (ms)"),
                   ("p99_resp_ms", "P99 Response Time (ms)")]
        names = [r["policy"] for r in basic]
        for ax_idx, (metric, title) in enumerate(metrics):
            ax = axes.flatten()[ax_idx]
            vals = [r[metric] for r in basic]
            colors = ["#2ca02c" if r["policy"] in ("LRU", "Decay", "LFU", "OPT") else "#1f77b4"
                      for r in basic]
            ax.bar(range(len(names)), vals, color=colors)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis="y")
        fig.suptitle("Part 3A: Policy Comparison (10GB Fast Cache)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "part3a_policy_comparison.png"), dpi=150)
        print(f"Saved part3a_policy_comparison.png")

    # Fig B: hybrid sweep lines
    hybrid = [r for r in all_rows if r["sub_experiment"] == "hybrid"]
    lru_row = next((r for r in basic if r["policy"] == "LRU"), None)
    if hybrid and lru_row:
        model_types = ["ML_Reg", "ML_Rank", "OPT_Imit_Reg", "OPT_Imit_Rank"]
        colors = {"ML_Reg": "#1f77b4", "ML_Rank": "#ff7f0e",
                  "OPT_Imit_Reg": "#2ca02c", "OPT_Imit_Rank": "#d62728"}
        pcts = [0, 10, 25, 50, 100]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax_idx, (metric, title) in enumerate([("hit_rate", "Hit Rate"),
                                                    ("mean_resp_ms", "Mean Resp (ms)")]):
            ax = axes[ax_idx]
            for mt in model_types:
                vals = []
                for p in pcts:
                    if p == 0:
                        vals.append(lru_row[metric])
                    elif p == 100:
                        ml_row = next((r for r in basic if r["policy"] == mt), None)
                        vals.append(ml_row[metric] if ml_row else None)
                    else:
                        h_row = next((r for r in hybrid
                                     if r["policy"] == f"Hybrid_{mt}_p{p}"), None)
                        vals.append(h_row[metric] if h_row else None)
                valid = [(x, v) for x, v in zip(pcts, vals) if v is not None]
                if valid:
                    ax.plot([x for x, v in valid], [v for x, v in valid], "o-",
                            label=mt.replace("_", " "), color=colors[mt], linewidth=2)
            ax.set_xlabel("ML Protection %")
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.suptitle("Part 3B: Hybrid LRU+ML Sweep (10GB Fast Cache)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "part3b_hybrid_sweep.png"), dpi=150)
        print(f"Saved part3b_hybrid_sweep.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["basic", "hybrid", "plot", "all"], default="all")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--ml-only", action="store_true", help="Skip heuristic policies")
    args = parser.parse_args()

    if args.batch is not None:
        _BATCH_ID = args.batch

    if args.phase == "plot":
        make_plots()
        sys.exit(0)

    ck = load_checkpoint()
    print("Loading eval data (shards 4-11, 4 saturation + 4 eval)...")
    reqs, warmup_n = load_eval_data()
    print(f"  {len(reqs)} requests, {warmup_n} warmup")

    if args.phase in ("basic", "all"):
        print("\n=== Part 3A: Basic Policy Comparison ===")
        run_basic_comparison(ck, reqs, warmup_n, ml_only=args.ml_only)

    if args.phase in ("hybrid", "all"):
        print("\n=== Part 3B: Hybrid LRU+ML Sweep ===")
        run_hybrid_sweep(ck, reqs, warmup_n)

    if args.phase == "all":
        make_plots()

    print("\nDone!")
