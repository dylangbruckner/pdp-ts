"""Part 2: Model size effect on ML models.
Sweep n_estimators and max_depth for each model type.
Uses optimal training size from Part 1 (4 shards for ML, TBD for OPT).
10GB fast cache, eval shards 4-5 with 50% warmup.
"""
import csv, json, os, sys, time, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np
import joblib

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PART_DIR = os.path.dirname(os.path.abspath(__file__))
PART1_DIR = os.path.join(os.path.dirname(PART_DIR), "part_1")
sys.path.insert(0, PROJECT)

from src.config import EvaluatorConfig, TierConfig
from src.evaluator import Evaluator
from src.trace_loader import load_thesios_csv
from src.ml_training import generate_training_data, train_model, train_ranking_model
from src.opt_training import generate_imitation_data, train_regressor, train_ranker
from src.policy import make_ml_policy, make_lru_policy
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

CSV_FIELDS = ["policy", "model_type", "n_estimators", "max_depth", "num_leaves",
              "hit_rate", "wa", "mean_resp_ms", "p99_resp_ms", "wall_s"]
RESULTS_CSV = os.path.join(RESULT_DIR, "part2_results.csv")

_BATCH_ID = None

def _ck_path():
    if _BATCH_ID is not None:
        return os.path.join(RESULT_DIR, f"checkpoint_batch{_BATCH_ID}.json")
    return os.path.join(RESULT_DIR, "checkpoint.json")

def _csv_path():
    if _BATCH_ID is not None:
        return os.path.join(RESULT_DIR, f"part2_batch{_BATCH_ID}.csv")
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
    reqs = load_shard_range(4, 6)
    warmup_n = len(reqs) // 2
    for r in reqs[:warmup_n]:
        r.is_warmup = True
    return reqs, warmup_n

def run_eval(name, pol, reqs, warmup_n, model_type="", n_est=0, depth=0, leaves=0):
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
        "policy": name, "model_type": model_type,
        "n_estimators": n_est, "max_depth": depth, "num_leaves": leaves,
        "hit_rate": round(s["hit_rate"], 6), "wa": round(wa, 4),
        "mean_resp_ms": round(s["mean_response_time_s"] * 1000, 4),
        "p99_resp_ms": round(s["p99_response_time_s"] * 1000, 4),
        "wall_s": round(elapsed, 1),
    }
    print(f"  {name:<45s} hit={row['hit_rate']:.4f}  mean={row['mean_resp_ms']:.3f}ms  ({elapsed:.0f}s)")
    return row

MODEL_SIZE_CONFIGS = [
    {"n_estimators": 50,   "max_depth": 3,  "num_leaves": 8},
    {"n_estimators": 100,  "max_depth": 4,  "num_leaves": 15},
    {"n_estimators": 200,  "max_depth": 6,  "num_leaves": 31},
    {"n_estimators": 500,  "max_depth": 6,  "num_leaves": 31},   # ML default
    {"n_estimators": 300,  "max_depth": 8,  "num_leaves": 63},   # OPT default
    {"n_estimators": 500,  "max_depth": 8,  "num_leaves": 63},
    {"n_estimators": 1000, "max_depth": 10, "num_leaves": 127},
]


def run_ml_reg_sweep(ck, reqs, warmup_n):
    """Train and eval ML Regressor at each model size. Uses 4 shards training data."""
    train_key = "ml_reg_train_data"
    data_path = os.path.join(MODEL_DIR, "ml_reg_train_data.npz")

    if not is_done(ck, train_key):
        print("Generating ML training data (4 shards)...")
        train_reqs = load_shard_range(0, 4)
        X, y, ts = generate_training_data(train_reqs, fast_capacity=FAST_CAP, return_timestamps=True)
        np.savez_compressed(data_path, X=X, y=y, ts=ts)
        mark_done(ck, train_key)
        print(f"  {X.shape[0]} samples saved")
    else:
        print("ML training data already generated.")

    data = np.load(data_path)
    X, y, ts = data["X"], data["y"], data["ts"]

    for cfg in MODEL_SIZE_CONFIGS:
        n_est, depth, leaves = cfg["n_estimators"], cfg["max_depth"], cfg["num_leaves"]
        name = f"ML_Reg_t{n_est}_d{depth}_l{leaves}"
        model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
        train_k = f"train_{name}"
        eval_k = f"eval_{name}"

        if not is_done(ck, train_k):
            print(f"  Training {name}...")
            params = {"n_estimators": n_est, "max_depth": depth, "num_leaves": leaves}
            train_model(X, y, model_path, params=params)
            mark_done(ck, train_k)

        if not is_done(ck, eval_k):
            pol = make_ml_policy(FAST_CAP, model_path, fast_fill_threshold=0.9)
            row = run_eval(name, pol, reqs, warmup_n,
                          model_type="ML_Reg", n_est=n_est, depth=depth, leaves=leaves)
            append_result(row)
            mark_done(ck, eval_k)
        else:
            print(f"  {name} already done.")


def run_ml_rank_sweep(ck, reqs, warmup_n):
    """Train and eval ML Ranker at each model size."""
    train_key = "ml_rank_train_data"
    data_path = os.path.join(MODEL_DIR, "ml_rank_train_data.npz")

    if not is_done(ck, train_key):
        print("Generating ML training data for ranker (4 shards)...")
        train_reqs = load_shard_range(0, 4)
        X, y, ts = generate_training_data(train_reqs, fast_capacity=FAST_CAP, return_timestamps=True)
        np.savez_compressed(data_path, X=X, y=y, ts=ts)
        mark_done(ck, train_key)
        print(f"  {X.shape[0]} samples saved")
    else:
        print("ML Rank training data already generated.")

    data = np.load(data_path)
    X, y, ts = data["X"], data["y"], data["ts"]

    for cfg in MODEL_SIZE_CONFIGS:
        n_est, depth, leaves = cfg["n_estimators"], cfg["max_depth"], cfg["num_leaves"]
        name = f"ML_Rank_t{n_est}_d{depth}_l{leaves}"
        model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
        train_k = f"train_{name}"
        eval_k = f"eval_{name}"

        if not is_done(ck, train_k):
            print(f"  Training {name}...")
            params = {"n_estimators": n_est, "max_depth": depth, "num_leaves": leaves}
            train_ranking_model(X, y, model_path, timestamps=ts, params=params)
            mark_done(ck, train_k)

        if not is_done(ck, eval_k):
            pol = make_ml_policy(FAST_CAP, model_path, fast_fill_threshold=0.9)
            row = run_eval(name, pol, reqs, warmup_n,
                          model_type="ML_Rank", n_est=n_est, depth=depth, leaves=leaves)
            append_result(row)
            mark_done(ck, eval_k)
        else:
            print(f"  {name} already done.")


def run_opt_reg_sweep(ck, reqs, warmup_n):
    """Train and eval OPT Imit Reg at each model size. Best training: 1 shard."""
    data_path = os.path.join(PART1_DIR, "models", "opt_orig_data_1shards.npz")
    if not os.path.exists(data_path):
        data_path = os.path.join(PART1_DIR, "models", "opt_oracle_data.npz")
    if not os.path.exists(data_path):
        print("  ERROR: No OPT training data found. Run Part 1 OPT retrain first.")
        return

    data = np.load(data_path)
    X, y, g = data["X"], data["y"], data["g"]
    print(f"OPT Imit Reg: loaded {X.shape[0]} samples (1 shard, optimal from Part 1)")

    for cfg in MODEL_SIZE_CONFIGS:
        n_est, depth, leaves = cfg["n_estimators"], cfg["max_depth"], cfg["num_leaves"]
        name = f"OPT_Reg_t{n_est}_d{depth}_l{leaves}"
        model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
        train_k = f"train_{name}"
        eval_k = f"eval_{name}"

        if not is_done(ck, train_k):
            print(f"  Training {name}...")
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(
                n_estimators=n_est, max_depth=depth, learning_rate=0.05,
                num_leaves=leaves, min_child_samples=20, subsample=0.8,
                colsample_bytree=0.8, verbose=-1,
            )
            model.fit(X, y)
            joblib.dump(model, model_path)
            mark_done(ck, train_k)

        if not is_done(ck, eval_k):
            pol = make_ml_policy(FAST_CAP, model_path, fast_fill_threshold=0.9)
            row = run_eval(name, pol, reqs, warmup_n,
                          model_type="OPT_Imit_Reg", n_est=n_est, depth=depth, leaves=leaves)
            append_result(row)
            mark_done(ck, eval_k)
        else:
            print(f"  {name} already done.")


def run_opt_rank_sweep(ck, reqs, warmup_n):
    """Train and eval OPT Imit Rank at each model size. Best training: 10k samples."""
    data_path = os.path.join(PART1_DIR, "models", "opt_orig_data_1shards.npz")
    if not os.path.exists(data_path):
        data_path = os.path.join(PART1_DIR, "models", "opt_oracle_data.npz")
    if not os.path.exists(data_path):
        print("  ERROR: No OPT training data found. Run Part 1 OPT retrain first.")
        return

    data = np.load(data_path)
    X, y, g = data["X"], data["y"], data["g"]
    # Best training size for OPT Rank is 10k from Part 1
    X, y, g = X[:10000], y[:10000], g[:10000]
    print(f"OPT Imit Rank: using {X.shape[0]} samples (10k, optimal from Part 1)")

    for cfg in MODEL_SIZE_CONFIGS:
        n_est, depth, leaves = cfg["n_estimators"], cfg["max_depth"], cfg["num_leaves"]
        name = f"OPT_Rank_t{n_est}_d{depth}_l{leaves}"
        model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
        train_k = f"train_{name}"
        eval_k = f"eval_{name}"

        if not is_done(ck, train_k):
            print(f"  Training {name}...")
            from lightgbm import LGBMRanker
            _, group_counts = np.unique(g, return_counts=True)
            y_raw = np.expm1(y)
            relevance = np.zeros(len(y_raw), dtype=np.int32)
            relevance[y_raw > 1] = 1
            relevance[y_raw > 10] = 2
            relevance[y_raw > 60] = 3
            relevance[y_raw > 300] = 4
            model = LGBMRanker(
                objective="lambdarank", n_estimators=n_est, max_depth=depth,
                learning_rate=0.05, num_leaves=leaves, min_child_samples=10,
                subsample=0.8, colsample_bytree=0.8, verbose=-1,
                label_gain=[0, 1, 3, 7, 15],
            )
            model.fit(X, relevance, group=group_counts)
            joblib.dump(model, model_path)
            mark_done(ck, train_k)

        if not is_done(ck, eval_k):
            pol = make_ml_policy(FAST_CAP, model_path, fast_fill_threshold=0.9)
            row = run_eval(name, pol, reqs, warmup_n,
                          model_type="OPT_Imit_Rank", n_est=n_est, depth=depth, leaves=leaves)
            append_result(row)
            mark_done(ck, eval_k)
        else:
            print(f"  {name} already done.")


def run_baselines(ck, reqs, warmup_n):
    for bname, make_fn in [("LRU", lambda: make_lru_policy(FAST_CAP)),
                            ("Decay", lambda: make_decay_policy(FAST_CAP, alpha=0.5, decay_rate=0.05, fast_fill_threshold=0.9))]:
        key = f"eval_baseline_{bname}"
        if is_done(ck, key):
            print(f"  {bname} already done.")
            continue
        pol = make_fn()
        row = run_eval(bname, pol, reqs, warmup_n, model_type="baseline")
        append_result(row)
        mark_done(ck, key)


def make_plots():
    import glob as _glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_rows = []
    seen = set()
    for f in sorted(_glob.glob(os.path.join(RESULT_DIR, "part2_batch*.csv"))):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                key = r["policy"]
                if key not in seen:
                    all_rows.append(r)
                    seen.add(key)
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
        r["n_estimators"] = int(r["n_estimators"])
        r["max_depth"] = int(r["max_depth"])

    baselines = {r["policy"]: r for r in all_rows if r["model_type"] == "baseline"}
    model_rows = [r for r in all_rows if r["model_type"] != "baseline"]

    model_types = ["ML_Reg", "ML_Rank", "OPT_Imit_Reg", "OPT_Imit_Rank"]
    colors = {"ML_Reg": "#1f77b4", "ML_Rank": "#ff7f0e",
              "OPT_Imit_Reg": "#2ca02c", "OPT_Imit_Rank": "#d62728"}

    metrics = [("hit_rate", "Hit Rate"), ("mean_resp_ms", "Mean Response Time (ms)")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (metric, title) in enumerate(metrics):
        ax = axes[ax_idx]
        for mt in model_types:
            mt_rows = sorted([r for r in model_rows if r["model_type"] == mt],
                           key=lambda r: (r["n_estimators"], r["max_depth"]))
            if not mt_rows:
                continue
            x_labels = [f"t{r['n_estimators']}\nd{r['max_depth']}" for r in mt_rows]
            vals = [r[metric] for r in mt_rows]
            ax.plot(range(len(vals)), vals, "o-", label=mt.replace("_", " "),
                    color=colors.get(mt, "gray"), linewidth=2, markersize=6)

        for bname, brow in baselines.items():
            ax.axhline(brow[metric], linestyle="--", color="gray", alpha=0.7, label=bname)

        ax.set_xticks(range(len(MODEL_SIZE_CONFIGS)))
        ax.set_xticklabels([f"t{c['n_estimators']}\nd{c['max_depth']}" for c in MODEL_SIZE_CONFIGS],
                          fontsize=7)
        ax.set_xlabel("Model Size (trees / depth)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Part 2: Model Size Effect on ML Models (10GB Fast Cache)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "part2_model_size_effect.png"), dpi=150)
    print(f"Saved figure to {FIG_DIR}/part2_model_size_effect.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", choices=["ml_reg", "ml_rank", "opt_reg", "opt_rank",
                                             "baselines", "all", "plot"],
                        default="all")
    parser.add_argument("--batch", type=int, help="Batch ID for parallel runs")
    args = parser.parse_args()

    if args.batch is not None:
        _BATCH_ID = args.batch

    if args.sweep == "plot":
        make_plots()
        sys.exit(0)

    ck = load_checkpoint()
    reqs, warmup_n = load_eval_data()
    print(f"Eval data: {len(reqs)} requests, {warmup_n} warmup")

    if args.sweep in ("baselines", "all"):
        run_baselines(ck, reqs, warmup_n)
    if args.sweep in ("ml_reg", "all"):
        run_ml_reg_sweep(ck, reqs, warmup_n)
    if args.sweep in ("ml_rank", "all"):
        run_ml_rank_sweep(ck, reqs, warmup_n)
    if args.sweep in ("opt_reg", "all"):
        run_opt_reg_sweep(ck, reqs, warmup_n)
    if args.sweep in ("opt_rank", "all"):
        run_opt_rank_sweep(ck, reqs, warmup_n)
    if args.sweep == "all":
        make_plots()

    print("\nDone!")
