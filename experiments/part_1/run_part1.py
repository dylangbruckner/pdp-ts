"""Part 1: Training size effect on ML models.
10GB fast cache, cluster_1_group_1 shards.
Training: shards 0-3 (variable amount), OPT oracle: shards 0-19
Saturation: shards 4-7, Evaluation: shards 8-11
"""
import csv, json, os, sys, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PART_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)
sys.path.insert(0, PART_DIR)

from src.config import EvaluatorConfig, TierConfig
from src.evaluator import Evaluator
from src.trace_loader import load_thesios_csv
from src.ml_training import generate_training_data, train_model, train_ranking_model
from src.opt_training import train_regressor, train_ranker
from src.policy import make_lru_policy, make_ml_policy, make_opt_policy
from src.decay_policy import make_decay_policy
from fast_opt_training import generate_imitation_data_fast

GB, MB = 1024**3, 1024**2
FAST_CAP = 10 * GB
TRACE_DIR = os.path.join(PROJECT, "traces/workload2/cluster_1_group_1")
SHARDS = sorted(os.listdir(TRACE_DIR))
MODEL_DIR = os.path.join(PART_DIR, "models")
RESULT_DIR = os.path.join(PART_DIR, "results")
FIG_DIR = os.path.join(PART_DIR, "figures")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

CHECKPOINT_FILE = os.path.join(RESULT_DIR, "checkpoint.json")
RESULTS_CSV = os.path.join(RESULT_DIR, "part1_results.csv")
_BATCH_ID = None  # set by --batch for per-batch checkpoints
CSV_FIELDS = ["policy", "train_size_label", "train_rows", "hit_rate", "wa",
              "mean_resp_ms", "p99_resp_ms", "wall_s"]

SLOW_CFG = TierConfig(200*MB, 150*MB, 5e-3, 5e-3, 8, 8)
FAST_CFG = TierConfig(3.5*GB, 3.0*GB, 20e-6, 20e-6, 32, 32, FAST_CAP)

TRAIN_SIZES = {
    "10k": 10_000,
    "100k": 100_000,
    "1shard": None,   # 1 shard
    "2shards": None,  # 2 shards
    "4shards": None,  # 4 shards
}
TRAIN_SHARD_COUNTS = {"10k": 1, "100k": 1, "1shard": 1, "2shards": 2, "4shards": 4}
TRAIN_ROW_LIMITS = {"10k": 10_000, "100k": 100_000, "1shard": None, "2shards": None, "4shards": None}

OPT_ORACLE_SHARDS = 10


def shard_path(idx):
    return os.path.join(TRACE_DIR, SHARDS[idx])


def _ck_path():
    if _BATCH_ID is not None:
        return os.path.join(RESULT_DIR, f"checkpoint_batch{_BATCH_ID}.json")
    return CHECKPOINT_FILE


def load_checkpoint():
    path = _ck_path()
    if os.path.exists(path):
        with open(path) as f:
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


def _csv_path():
    if _BATCH_ID is not None:
        return os.path.join(RESULT_DIR, f"part1_batch{_BATCH_ID}.csv")
    return RESULTS_CSV


def append_result(row):
    path = _csv_path()
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def load_shard_range(start, end):
    reqs = []
    for i in range(start, end):
        reqs.extend(load_thesios_csv(shard_path(i)))
    reqs.sort(key=lambda r: r.arrival_time)
    return reqs


EVAL_SHARD_RANGE = (4, 6)  # shards 4-5 for evaluation
WARMUP_FRAC = 0.5          # first 50% as warmup

def load_eval_data():
    """Load eval shards with first half as warmup (same-population split)."""
    reqs = load_shard_range(*EVAL_SHARD_RANGE)
    warmup_n = int(len(reqs) * WARMUP_FRAC)
    for r in reqs[:warmup_n]:
        r.is_warmup = True
    return reqs, warmup_n


def run_eval(name, pol, reqs, warmup_n, train_label="", train_rows=0):
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
        "policy": name,
        "train_size_label": train_label,
        "train_rows": train_rows,
        "hit_rate": round(s["hit_rate"], 6),
        "wa": round(wa, 4),
        "mean_resp_ms": round(s["mean_response_time_s"] * 1000, 4),
        "p99_resp_ms": round(s["p99_response_time_s"] * 1000, 4),
        "wall_s": round(elapsed, 1),
    }
    print(f"  {name:<40s} hit={row['hit_rate']:.4f}  WA={row['wa']:.3f}  "
          f"mean={row['mean_resp_ms']:.3f}ms  p99={row['p99_resp_ms']:.3f}ms  ({elapsed:.0f}s)")
    return row


# ── Phase 1: Generate OPT oracle data ──────────────────────────────

def generate_opt_oracle_data(ck):
    key = "opt_oracle_data"
    npz_path = os.path.join(MODEL_DIR, "opt_oracle_data.npz")
    if is_done(ck, key) and os.path.exists(npz_path):
        print("OPT oracle data already generated, skipping.")
        return

    print(f"Loading {OPT_ORACLE_SHARDS} shards for OPT oracle...")
    all_reqs = load_shard_range(0, OPT_ORACLE_SHARDS)
    print(f"  {len(all_reqs)} total requests for oracle visibility")

    train_reqs = load_shard_range(0, 4)
    print(f"  {len(train_reqs)} training simulation requests (shards 0-3)")

    print("Generating OPT imitation data (10GB fast, heap-optimized)...")
    X, y, g = generate_imitation_data_fast(
        train_reqs, fast_capacity=FAST_CAP,
        future_requests=all_reqs, sample_rate=1,
    )
    print(f"  {X.shape[0]} samples, {len(np.unique(g))} groups")
    print(f"  y stats: min={y.min():.2f} max={y.max():.2f} mean={y.mean():.2f}")

    np.savez_compressed(npz_path, X=X, y=y, g=g)
    mark_done(ck, key)
    print(f"  Saved to {npz_path}")


# ── Phase 2: Train all models ──────────────────────────────────────

def train_ml_models(ck):
    """Train ML Regressor and ML Ranker for each training size."""
    for label in TRAIN_SIZES:
        n_shards = TRAIN_SHARD_COUNTS[label]
        max_rows = TRAIN_ROW_LIMITS[label]

        reg_key = f"train_ml_reg_{label}"
        rank_key = f"train_ml_rank_{label}"
        reg_path = os.path.join(MODEL_DIR, f"ml_reg_{label}.joblib")
        rank_path = os.path.join(MODEL_DIR, f"ml_rank_{label}.joblib")

        if is_done(ck, reg_key) and is_done(ck, rank_key):
            print(f"ML models for {label} already trained, skipping.")
            continue

        print(f"\nTraining ML models for {label} (shards 0-{n_shards-1}, max_rows={max_rows})...")
        reqs = load_shard_range(0, n_shards)
        if max_rows and len(reqs) > max_rows:
            reqs = reqs[:max_rows]
        print(f"  Using {len(reqs)} requests")

        X, y, ts = generate_training_data(
            reqs, fast_capacity=FAST_CAP, return_timestamps=True,
        )
        print(f"  Training data: {X.shape[0]} samples")

        if not is_done(ck, reg_key):
            train_model(X, y, reg_path)
            mark_done(ck, reg_key)

        if not is_done(ck, rank_key):
            train_ranking_model(X, y, rank_path, timestamps=ts)
            mark_done(ck, rank_key)


def train_opt_imit_models(ck):
    """Train OPT Imit Regressor and Ranker for each training size."""
    npz_path = os.path.join(MODEL_DIR, "opt_oracle_data.npz")
    data = np.load(npz_path)
    X_all, y_all, g_all = data["X"], data["y"], data["g"]
    total = len(y_all)
    print(f"\nOPT imitation data: {total} samples total")

    for label in TRAIN_SIZES:
        reg_key = f"train_opt_reg_{label}"
        rank_key = f"train_opt_rank_{label}"
        reg_path = os.path.join(MODEL_DIR, f"opt_reg_{label}.joblib")
        rank_path = os.path.join(MODEL_DIR, f"opt_rank_{label}.joblib")

        if is_done(ck, reg_key) and is_done(ck, rank_key):
            print(f"OPT imit models for {label} already trained, skipping.")
            continue

        n_shards = TRAIN_SHARD_COUNTS[label]
        max_rows = TRAIN_ROW_LIMITS[label]

        if max_rows:
            n_train = min(max_rows, total)
        else:
            frac = n_shards / 4.0
            n_train = int(total * frac)

        X, y, g = X_all[:n_train], y_all[:n_train], g_all[:n_train]
        print(f"\nTraining OPT imit models for {label} ({n_train} samples)...")

        if not is_done(ck, reg_key):
            train_regressor(X, y, reg_path)
            mark_done(ck, reg_key)

        if not is_done(ck, rank_key):
            train_ranker(X, y, g, rank_path)
            mark_done(ck, rank_key)


# ── Phase 3: Evaluate ──────────────────────────────────────────────

def run_baselines(ck, reqs, warmup_n):
    """Run LRU, Decay, OPT baselines (once each)."""
    baselines = [
        ("LRU", lambda: make_lru_policy(FAST_CAP)),
        ("Decay", lambda: make_decay_policy(FAST_CAP, alpha=0.5, decay_rate=0.05,
                                             fast_fill_threshold=0.9)),
        ("OPT", lambda: make_opt_policy(FAST_CAP, reqs, fast_fill_threshold=0.9,
                                         background_interval=5.0)),
    ]
    for name, make_fn in baselines:
        key = f"eval_baseline_{name}"
        if is_done(ck, key):
            print(f"  {name} baseline already done, skipping.")
            continue
        pol = make_fn()
        row = run_eval(name, pol, reqs, warmup_n, train_label="baseline", train_rows=0)
        append_result(row)
        mark_done(ck, key)


def run_ml_evals(ck, reqs, warmup_n):
    """Run all ML model evaluations."""
    for label in TRAIN_SIZES:
        n_shards = TRAIN_SHARD_COUNTS[label]
        max_rows = TRAIN_ROW_LIMITS[label]
        if max_rows:
            approx_rows = max_rows
        else:
            approx_rows = n_shards * 168000

        models = [
            (f"ML_Reg_{label}", os.path.join(MODEL_DIR, f"ml_reg_{label}.joblib")),
            (f"ML_Rank_{label}", os.path.join(MODEL_DIR, f"ml_rank_{label}.joblib")),
            (f"OPT_Imit_Reg_{label}", os.path.join(MODEL_DIR, f"opt_reg_{label}.joblib")),
            (f"OPT_Imit_Rank_{label}", os.path.join(MODEL_DIR, f"opt_rank_{label}.joblib")),
        ]
        for name, model_path in models:
            key = f"eval_{name}"
            if is_done(ck, key):
                print(f"  {name} already evaluated, skipping.")
                continue
            if not os.path.exists(model_path):
                print(f"  WARNING: {model_path} not found, skipping {name}")
                continue
            pol = make_ml_policy(FAST_CAP, model_path, fast_fill_threshold=0.9)
            row = run_eval(name, pol, reqs, warmup_n,
                          train_label=label, train_rows=approx_rows)
            append_result(row)
            mark_done(ck, key)


# ── Phase 4: Plot ──────────────────────────────────────────────────

def merge_batch_csvs():
    """Merge per-batch CSVs into the main results CSV."""
    all_rows = []
    seen = set()
    for f in sorted(Glob_pattern(os.path.join(RESULT_DIR, "part1_batch*.csv"))):
        with open(f) as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                key = (r["policy"], r["train_size_label"])
                if key not in seen:
                    all_rows.append(r)
                    seen.add(key)
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV) as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                key = (r["policy"], r["train_size_label"])
                if key not in seen:
                    all_rows.append(r)
                    seen.add(key)
    with open(RESULTS_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Merged {len(all_rows)} results into {RESULTS_CSV}")


def Glob_pattern(pattern):
    import glob as _glob
    return _glob.glob(pattern)


def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    merge_batch_csvs()
    rows = []
    with open(RESULTS_CSV) as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["hit_rate"] = float(r["hit_rate"])
            r["wa"] = float(r["wa"])
            r["mean_resp_ms"] = float(r["mean_resp_ms"])
            r["p99_resp_ms"] = float(r["p99_resp_ms"])
            r["train_rows"] = int(r["train_rows"])
            rows.append(r)

    baselines = {r["policy"]: r for r in rows if r["train_size_label"] == "baseline"}
    ml_rows = [r for r in rows if r["train_size_label"] != "baseline"]

    model_types = ["ML_Reg", "ML_Rank", "OPT_Imit_Reg", "OPT_Imit_Rank"]
    labels_ordered = ["10k", "100k", "1shard", "2shards", "4shards"]
    x_labels = ["10K", "100K", "1 shard", "2 shards", "4 shards"]
    metrics = [("hit_rate", "Hit Rate"), ("wa", "Write Amplification"),
               ("mean_resp_ms", "Mean Response Time (ms)"),
               ("p99_resp_ms", "P99 Response Time (ms)")]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = {"ML_Reg": "#1f77b4", "ML_Rank": "#ff7f0e",
              "OPT_Imit_Reg": "#2ca02c", "OPT_Imit_Rank": "#d62728"}
    display_names = {"ML_Reg": "ML Regressor", "ML_Rank": "ML Ranker",
                     "OPT_Imit_Reg": "OPT Imit Reg", "OPT_Imit_Rank": "OPT Imit Rank"}

    for ax_idx, (metric, title) in enumerate(metrics):
        ax = axes[ax_idx]
        x_pos = range(len(labels_ordered))

        for mt in model_types:
            vals = []
            for lab in labels_ordered:
                name = f"{mt}_{lab}"
                match = [r for r in ml_rows if r["policy"] == name]
                vals.append(match[0][metric] if match else None)
            valid_x = [x for x, v in zip(x_pos, vals) if v is not None]
            valid_v = [v for v in vals if v is not None]
            ax.plot(valid_x, valid_v, "o-", label=display_names[mt],
                    color=colors[mt], linewidth=2, markersize=6)

        for bname, brow in baselines.items():
            style = "--" if bname != "OPT" else ":"
            ax.axhline(brow[metric], linestyle=style, color="gray",
                       alpha=0.7, label=bname)

        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Training Size")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Part 1: Training Size Effect on ML Models (10GB Fast Cache)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "part1_training_size_effect.png"), dpi=150)
    print(f"Saved figure to {FIG_DIR}/part1_training_size_effect.png")


# ── Targeted evaluation: run only specific models ──────────────────

def run_targeted_evals(ck, reqs, warmup_n, targets):
    """Run specific evaluations by key. targets is a list of eval keys."""
    for key in targets:
        if is_done(ck, key):
            print(f"  {key} already done, skipping.")
            continue
        if key.startswith("eval_baseline_"):
            name = key.replace("eval_baseline_", "")
            if name == "LRU":
                pol = make_lru_policy(FAST_CAP)
            elif name == "Decay":
                pol = make_decay_policy(FAST_CAP, alpha=0.5, decay_rate=0.05,
                                         fast_fill_threshold=0.9)
            elif name == "OPT":
                pol = make_opt_policy(FAST_CAP, reqs, fast_fill_threshold=0.9,
                                       background_interval=5.0)
            else:
                print(f"  Unknown baseline: {name}")
                continue
            row = run_eval(name, pol, reqs, warmup_n, train_label="baseline", train_rows=0)
            append_result(row)
            mark_done(ck, key)
        elif key.startswith("eval_"):
            model_name = key.replace("eval_", "")
            parts = model_name.rsplit("_", 1)
            if len(parts) == 2:
                # e.g. ML_Reg_10k -> prefix=ML_Reg, label=10k
                # but some have 3 parts: OPT_Imit_Reg_10k
                pass
            for label in TRAIN_SIZES:
                if model_name.endswith(f"_{label}"):
                    prefix = model_name[: -len(f"_{label}")]
                    break
            else:
                print(f"  Can't parse model name from key: {key}")
                continue

            model_file_map = {
                "ML_Reg": f"ml_reg_{label}.joblib",
                "ML_Rank": f"ml_rank_{label}.joblib",
                "OPT_Imit_Reg": f"opt_reg_{label}.joblib",
                "OPT_Imit_Rank": f"opt_rank_{label}.joblib",
            }
            model_file = model_file_map.get(prefix)
            if not model_file:
                print(f"  Unknown model prefix: {prefix}")
                continue
            model_path = os.path.join(MODEL_DIR, model_file)
            if not os.path.exists(model_path):
                print(f"  Model not found: {model_path}")
                continue
            n_shards = TRAIN_SHARD_COUNTS[label]
            max_rows = TRAIN_ROW_LIMITS[label]
            approx_rows = max_rows if max_rows else n_shards * 168000
            pol = make_ml_policy(FAST_CAP, model_path, fast_fill_threshold=0.9)
            row = run_eval(model_name, pol, reqs, warmup_n,
                          train_label=label, train_rows=approx_rows)
            append_result(row)
            mark_done(ck, key)


# ── Main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["oracle", "train", "eval", "plot", "all",
                                             "eval-batch"], default="all")
    parser.add_argument("--batch", type=int, help="Eval batch number (0-4) for parallel runs")
    parser.add_argument("--targets", nargs="*", help="Specific eval keys to run")
    args = parser.parse_args()

    if args.batch is not None:
        _BATCH_ID = args.batch

    ck = load_checkpoint()

    if args.phase in ("oracle", "all"):
        print("=" * 60)
        print("Phase 1: Generate OPT oracle data")
        print("=" * 60)
        generate_opt_oracle_data(ck)

    if args.phase in ("train", "all"):
        print("\n" + "=" * 60)
        print("Phase 2: Train all models")
        print("=" * 60)
        train_ml_models(ck)
        train_opt_imit_models(ck)

    if args.phase in ("eval", "all"):
        print("\n" + "=" * 60)
        print("Phase 3: Evaluate")
        print("=" * 60)
        print("Loading evaluation data (shards 4-11)...")
        reqs, warmup_n = load_eval_data()
        print(f"  {len(reqs)} requests, {warmup_n} warmup")
        run_baselines(ck, reqs, warmup_n)
        run_ml_evals(ck, reqs, warmup_n)

    if args.phase == "eval-batch" and args.batch is not None:
        print(f"\nRunning eval batch {args.batch}...")
        reqs, warmup_n = load_eval_data()
        print(f"  {len(reqs)} requests, {warmup_n} warmup")

        all_eval_keys = []
        all_eval_keys.extend([f"eval_baseline_{b}" for b in ["LRU", "Decay", "OPT"]])
        for label in ["10k", "100k", "1shard", "2shards", "4shards"]:
            for prefix in ["ML_Reg", "ML_Rank", "OPT_Imit_Reg", "OPT_Imit_Rank"]:
                all_eval_keys.append(f"eval_{prefix}_{label}")

        batch_size = (len(all_eval_keys) + 3) // 4
        start = args.batch * batch_size
        end = min(start + batch_size, len(all_eval_keys))
        batch_keys = all_eval_keys[start:end]
        print(f"  Batch {args.batch}: {len(batch_keys)} evaluations")
        for k in batch_keys:
            print(f"    - {k}")
        run_targeted_evals(ck, reqs, warmup_n, batch_keys)

    if args.phase in ("plot", "all"):
        print("\n" + "=" * 60)
        print("Phase 4: Plot")
        print("=" * 60)
        make_plots()

    if args.phase == "eval-batch" and args.targets:
        print(f"\nRunning targeted evals: {args.targets}")
        reqs, warmup_n = load_eval_data()
        run_targeted_evals(ck, reqs, warmup_n, args.targets)

    print("\nDone!")
