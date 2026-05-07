"""Part 5: Policy comparison across cache sizes (512MB to 100GB).
Models retrained at each cache size. Eval shards proportional to cache size.
Training: shards 0-3. Eval: shards 4+, 50% warmup.
Includes best 2 hybrids (ML_Reg p25, OPT_Rank p10).
"""
import csv, json, math, os, sys, time, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np
import joblib

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PART_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)

from src.config import EvaluatorConfig, TierConfig
from src.evaluator import Evaluator
from src.trace_loader import load_thesios_csv
from src.ml_training import generate_training_data, train_model, train_ranking_model
from src.opt_training import generate_imitation_data, train_regressor, train_ranker
from src.policy import make_lru_policy, make_ml_policy, make_hybrid_policy
from src.decay_policy import make_decay_policy

GB, MB = 1024**3, 1024**2
WS_PER_SHARD = 65 * GB
TRACE_DIR = os.path.join(PROJECT, "traces/workload2/cluster_1_group_1")
SHARDS = sorted(os.listdir(TRACE_DIR))
MODEL_DIR = os.path.join(PART_DIR, "models")
RESULT_DIR = os.path.join(PART_DIR, "results")
FIG_DIR = os.path.join(PART_DIR, "figures")
for d in [MODEL_DIR, RESULT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

SLOW_CFG = TierConfig(200*MB, 150*MB, 5e-3, 5e-3, 8, 8)

CSV_FIELDS = ["policy", "cache_size_label", "cache_bytes", "n_eval_shards",
              "hit_rate", "wa", "mean_resp_ms", "p99_resp_ms", "wall_s"]
RESULTS_CSV = os.path.join(RESULT_DIR, "part5_results.csv")
CHECKPOINT = os.path.join(RESULT_DIR, "checkpoint.json")

CACHE_SIZES = [
    ("512MB",  512 * MB),
    ("1GB",    1 * GB),
    ("2GB",    2 * GB),
    ("5GB",    5 * GB),
    ("10GB",   10 * GB),
    ("20GB",   20 * GB),
    ("100GB",  100 * GB),
]

TRAIN_SHARDS = 4
EVAL_START = 4
MAX_EVAL_SHARD = len(SHARDS)

def eval_shards_for_size(cap):
    """Scale eval shards so cache is ~8-12% of working set. Min 2, max available."""
    target_ws = cap / 0.10
    n = max(2, math.ceil(target_ws / WS_PER_SHARD))
    n = min(n, MAX_EVAL_SHARD - EVAL_START)
    return n

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

def load_shard_range(start, end):
    reqs = []
    for i in range(start, min(end, len(SHARDS))):
        reqs.extend(load_thesios_csv(shard_path(i)))
    reqs.sort(key=lambda r: r.arrival_time)
    return reqs

def make_fast_cfg(cap):
    return TierConfig(3.5*GB, 3.0*GB, 20e-6, 20e-6, 32, 32, cap)

def load_eval_data(cap):
    n_shards = eval_shards_for_size(cap)
    end = min(EVAL_START + n_shards, MAX_EVAL_SHARD)
    reqs = load_shard_range(EVAL_START, end)
    warmup_n = len(reqs) // 2
    for r in reqs[:warmup_n]:
        r.is_warmup = True
    return reqs, warmup_n, n_shards

def run_eval(name, pol, reqs, warmup_n, cap, size_label, n_shards):
    fast_cfg = make_fast_cfg(cap)
    cfg = EvaluatorConfig(
        fast_cfg, SLOW_CFG, shard_path(0), "/dev/null",
        warmup_ops=warmup_n, always_write_slow=True, on_fast_full="spill",
    )
    t0 = time.perf_counter()
    mc = Evaluator(cfg, pol, reqs).run()
    elapsed = time.perf_counter() - t0
    s = mc.global_metrics.summary()
    wa = mc.global_metrics.write_amplification(mc._requested_write_bytes)
    row = {
        "policy": name, "cache_size_label": size_label, "cache_bytes": cap,
        "n_eval_shards": n_shards,
        "hit_rate": round(s["hit_rate"], 6), "wa": round(wa, 4),
        "mean_resp_ms": round(s["mean_response_time_s"] * 1000, 4),
        "p99_resp_ms": round(s["p99_response_time_s"] * 1000, 4),
        "wall_s": round(elapsed, 1),
    }
    print(f"  {name:<30s} @{size_label:<6s} ({n_shards}sh) hit={row['hit_rate']:.4f}  mean={row['mean_resp_ms']:.3f}ms  ({elapsed:.0f}s)")
    return row


def train_models_for_size(ck, cap, size_label):
    train_reqs = load_shard_range(0, TRAIN_SHARDS)

    ml_data_key = f"ml_data_{size_label}"
    ml_data_path = os.path.join(MODEL_DIR, f"ml_data_{size_label}.npz")
    if not is_done(ck, ml_data_key):
        print(f"  Generating ML training data at {size_label}...")
        X, y, ts = generate_training_data(train_reqs, fast_capacity=cap, return_timestamps=True)
        np.savez_compressed(ml_data_path, X=X, y=y, ts=ts)
        mark_done(ck, ml_data_key)
        print(f"    {X.shape[0]} samples")

    ml_data = np.load(ml_data_path)
    X_ml, y_ml, ts_ml = ml_data["X"], ml_data["y"], ml_data["ts"]

    for mtype, train_fn in [
        ("ml_reg", lambda X, y, p: train_model(X, y, p, params={"n_estimators": 1000, "max_depth": 10, "num_leaves": 127})),
        ("ml_rank", lambda X, y, p: train_ranking_model(X, y, p, timestamps=ts_ml, params={"n_estimators": 500, "max_depth": 8, "num_leaves": 63})),
    ]:
        key = f"train_{mtype}_{size_label}"
        path = os.path.join(MODEL_DIR, f"{mtype}_{size_label}.joblib")
        if not is_done(ck, key):
            print(f"  Training {mtype} at {size_label}...")
            train_fn(X_ml, y_ml, path)
            mark_done(ck, key)

    opt_data_key = f"opt_data_{size_label}"
    opt_data_path = os.path.join(MODEL_DIR, f"opt_data_{size_label}.npz")
    if not is_done(ck, opt_data_key):
        print(f"  Generating OPT imitation data at {size_label}...")
        opt_reqs = load_shard_range(0, 1)
        X_opt, y_opt, g_opt = generate_imitation_data(opt_reqs, fast_capacity=cap, sample_rate=2)
        if len(y_opt) == 0:
            print(f"    0 samples from 1 shard, trying 4...")
            X_opt, y_opt, g_opt = generate_imitation_data(train_reqs, fast_capacity=cap, sample_rate=2)
        np.savez_compressed(opt_data_path, X=X_opt, y=y_opt, g=g_opt)
        mark_done(ck, opt_data_key)
        print(f"    {X_opt.shape[0]} samples")

    opt_data = np.load(opt_data_path)
    X_o, y_o, g_o = opt_data["X"], opt_data["y"], opt_data["g"]

    if len(y_o) > 0:
        key = f"train_opt_reg_{size_label}"
        path = os.path.join(MODEL_DIR, f"opt_reg_{size_label}.joblib")
        if not is_done(ck, key):
            print(f"  Training OPT_Imit_Reg at {size_label}...")
            train_regressor(X_o, y_o, path)
            mark_done(ck, key)

        key = f"train_opt_rank_{size_label}"
        path = os.path.join(MODEL_DIR, f"opt_rank_{size_label}.joblib")
        if not is_done(ck, key):
            n = min(10000, len(y_o))
            print(f"  Training OPT_Imit_Rank at {size_label} ({n} samples)...")
            from lightgbm import LGBMRanker
            _, gc = np.unique(g_o[:n], return_counts=True)
            yr = np.expm1(y_o[:n])
            rel = np.zeros(n, dtype=np.int32)
            rel[yr > 1] = 1; rel[yr > 10] = 2; rel[yr > 60] = 3; rel[yr > 300] = 4
            m = LGBMRanker(objective="lambdarank", n_estimators=100, max_depth=4,
                           learning_rate=0.05, num_leaves=15, min_child_samples=10,
                           subsample=0.8, colsample_bytree=0.8, verbose=-1,
                           label_gain=[0, 1, 3, 7, 15])
            m.fit(X_o[:n], rel, group=gc)
            joblib.dump(m, path)
            mark_done(ck, key)


def run_size(ck, size_label, cap):
    n_shards = eval_shards_for_size(cap)
    print(f"\n{'='*60}")
    print(f"Cache size: {size_label} — {n_shards} eval shards (cache/WS = {cap/(n_shards*WS_PER_SHARD)*100:.1f}%)")
    print(f"{'='*60}")

    train_models_for_size(ck, cap, size_label)

    reqs, warmup_n, n_sh = load_eval_data(cap)
    print(f"  Eval: {len(reqs)} requests, {warmup_n} warmup, {n_sh} shards")

    heuristics = [
        ("LRU", lambda: make_lru_policy(cap)),
        ("Decay", lambda: make_decay_policy(cap, alpha=0.5, decay_rate=0.05, fast_fill_threshold=0.9)),
    ]
    ml_models = [
        ("ML_Reg", os.path.join(MODEL_DIR, f"ml_reg_{size_label}.joblib")),
        ("ML_Rank", os.path.join(MODEL_DIR, f"ml_rank_{size_label}.joblib")),
        ("OPT_Imit_Reg", os.path.join(MODEL_DIR, f"opt_reg_{size_label}.joblib")),
        ("OPT_Imit_Rank", os.path.join(MODEL_DIR, f"opt_rank_{size_label}.joblib")),
    ]
    hybrids = [
        ("Hybrid_ML_Reg_p25", os.path.join(MODEL_DIR, f"ml_reg_{size_label}.joblib"), 0.25),
        ("Hybrid_OPT_Rank_p10", os.path.join(MODEL_DIR, f"opt_rank_{size_label}.joblib"), 0.10),
    ]

    for name, make_fn in heuristics:
        key = f"eval_{name}_{size_label}"
        if is_done(ck, key):
            print(f"  {name} @{size_label} done.")
            continue
        pol = make_fn()
        row = run_eval(name, pol, reqs, warmup_n, cap, size_label, n_sh)
        append_result(row)
        mark_done(ck, key)

    for name, model_path in ml_models:
        key = f"eval_{name}_{size_label}"
        if is_done(ck, key):
            print(f"  {name} @{size_label} done.")
            continue
        if not os.path.exists(model_path):
            print(f"  {name} @{size_label}: no model, skipping")
            continue
        pol = make_ml_policy(cap, model_path, fast_fill_threshold=0.9)
        row = run_eval(name, pol, reqs, warmup_n, cap, size_label, n_sh)
        append_result(row)
        mark_done(ck, key)

    for name, model_path, pct in hybrids:
        key = f"eval_{name}_{size_label}"
        if is_done(ck, key):
            print(f"  {name} @{size_label} done.")
            continue
        if not os.path.exists(model_path):
            print(f"  {name} @{size_label}: no model, skipping")
            continue
        pol = make_hybrid_policy(cap, model_path, protect_percentile=pct, fast_fill_threshold=0.9)
        row = run_eval(name, pol, reqs, warmup_n, cap, size_label, n_sh)
        append_result(row)
        mark_done(ck, key)


def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV) as f:
            for r in csv.DictReader(f):
                for k in ["hit_rate", "wa", "mean_resp_ms", "p99_resp_ms"]:
                    r[k] = float(r[k])
                rows.append(r)
    if not rows:
        print("No results to plot")
        return

    size_order = [s[0] for s in CACHE_SIZES]
    policies = ["LRU", "Decay", "ML_Reg", "ML_Rank", "OPT_Imit_Reg", "OPT_Imit_Rank",
                "Hybrid_ML_Reg_p25", "Hybrid_OPT_Rank_p10"]
    colors = {"LRU": "#333333", "Decay": "#666666",
              "ML_Reg": "#1f77b4", "ML_Rank": "#ff7f0e",
              "OPT_Imit_Reg": "#2ca02c", "OPT_Imit_Rank": "#d62728",
              "Hybrid_ML_Reg_p25": "#9467bd", "Hybrid_OPT_Rank_p10": "#e377c2"}
    styles = {"Hybrid_ML_Reg_p25": "--", "Hybrid_OPT_Rank_p10": "--"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [("hit_rate", "Hit Rate"), ("wa", "Write Amplification"),
               ("mean_resp_ms", "Mean Response Time (ms)"),
               ("p99_resp_ms", "P99 Response Time (ms)")]

    for ax_idx, (metric, title) in enumerate(metrics):
        ax = axes.flatten()[ax_idx]
        for pol in policies:
            vals, x_valid = [], []
            for i, sl in enumerate(size_order):
                match = [r for r in rows if r["policy"] == pol and r["cache_size_label"] == sl]
                if match:
                    vals.append(match[0][metric])
                    x_valid.append(i)
            if vals:
                ls = styles.get(pol, "-")
                ax.plot(x_valid, vals, f"o{ls}", label=pol, color=colors.get(pol, "gray"),
                        linewidth=2, markersize=6)
        ax.set_xticks(range(len(size_order)))
        ax.set_xticklabels(size_order, fontsize=8)
        ax.set_xlabel("Cache Size")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Part 5: Policy Comparison Across Cache Sizes", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "part5_cache_scaling.png"), dpi=150)
    print(f"Saved part5_cache_scaling.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", help="Run specific cache size (e.g. '10GB')")
    parser.add_argument("--phase", choices=["all", "plot"], default="all")
    args = parser.parse_args()

    if args.phase == "plot":
        make_plots()
        sys.exit(0)

    ck = load_checkpoint()

    # Print shard plan
    print("Eval shard plan:")
    for label, cap in CACHE_SIZES:
        n = eval_shards_for_size(cap)
        ratio = cap / (n * WS_PER_SHARD) * 100
        print(f"  {label:>6s}: {n:2d} shards, cache/WS = {ratio:.1f}%")
    print()

    if args.size:
        sizes = [(l, c) for l, c in CACHE_SIZES if l == args.size]
        if not sizes:
            print(f"Unknown size: {args.size}")
            sys.exit(1)
    else:
        sizes = CACHE_SIZES

    for label, cap in sizes:
        run_size(ck, label, cap)

    print("\nDone!")
