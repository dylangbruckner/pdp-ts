"""Retrain OPT Imit models using original generate_imitation_data (not fast version).
Generates data per shard count, trains reg+rank at each size, then evaluates all.
"""
import csv, json, os, sys, time, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np
import joblib

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PART_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)

from src.config import EvaluatorConfig, TierConfig
from src.evaluator import Evaluator
from src.trace_loader import load_thesios_csv
from src.opt_training import generate_imitation_data, train_regressor, train_ranker
from src.policy import make_ml_policy, make_lru_policy

GB, MB = 1024**3, 1024**2
FAST_CAP = 10 * GB
TRACE_DIR = os.path.join(PROJECT, "traces/workload2/cluster_1_group_1")
SHARDS = sorted(os.listdir(TRACE_DIR))
MODEL_DIR = os.path.join(PART_DIR, "models")
RESULT_DIR = os.path.join(PART_DIR, "results")

SLOW_CFG = TierConfig(200*MB, 150*MB, 5e-3, 5e-3, 8, 8)
FAST_CFG = TierConfig(3.5*GB, 3.0*GB, 20e-6, 20e-6, 32, 32, FAST_CAP)

CSV_FIELDS = ["policy", "train_size_label", "train_rows", "hit_rate", "wa",
              "mean_resp_ms", "p99_resp_ms", "wall_s"]

CHECKPOINT_FILE = os.path.join(RESULT_DIR, "checkpoint_opt_retrain.json")
RESULTS_CSV = os.path.join(RESULT_DIR, "part1_opt_retrain.csv")

def shard_path(idx):
    return os.path.join(TRACE_DIR, SHARDS[idx])

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed": []}

def save_checkpoint(ck):
    with open(CHECKPOINT_FILE, "w") as f:
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
        "policy": name, "train_size_label": train_label, "train_rows": train_rows,
        "hit_rate": round(s["hit_rate"], 6), "wa": round(wa, 4),
        "mean_resp_ms": round(s["mean_response_time_s"] * 1000, 4),
        "p99_resp_ms": round(s["p99_response_time_s"] * 1000, 4),
        "wall_s": round(elapsed, 1),
    }
    print(f"  {name:<40s} hit={row['hit_rate']:.4f}  WA={row['wa']:.3f}  "
          f"mean={row['mean_resp_ms']:.3f}ms  ({elapsed:.0f}s)")
    return row


TRAIN_CONFIGS = {
    "10k":     (1, 10_000),
    "100k":    (1, 100_000),
    "1shard":  (1, None),
    "2shards": (2, None),
    "4shards": (4, None),
}

def generate_and_train(ck, model_type):
    """model_type is 'reg' or 'rank'."""
    for label, (n_shards, max_rows) in TRAIN_CONFIGS.items():
        data_key = f"opt_data_{n_shards}shards"
        npz_path = os.path.join(MODEL_DIR, f"opt_orig_data_{n_shards}shards.npz")

        if not is_done(ck, data_key):
            print(f"\nGenerating OPT imitation data ({n_shards} shards, original generator)...")
            train_reqs = load_shard_range(0, n_shards)
            print(f"  {len(train_reqs)} training requests")
            t0 = time.perf_counter()
            X, y, g = generate_imitation_data(
                train_reqs, fast_capacity=FAST_CAP, sample_rate=2,
            )
            elapsed = time.perf_counter() - t0
            print(f"  {X.shape[0]} samples, {len(np.unique(g))} groups ({elapsed:.0f}s)")
            np.savez_compressed(npz_path, X=X, y=y, g=g)
            mark_done(ck, data_key)
        else:
            print(f"  OPT data for {n_shards} shards already exists.")

        data = np.load(npz_path)
        X_all, y_all, g_all = data["X"], data["y"], data["g"]

        if max_rows and len(y_all) > max_rows:
            X, y, g = X_all[:max_rows], y_all[:max_rows], g_all[:max_rows]
        else:
            X, y, g = X_all, y_all, g_all

        model_key = f"train_opt_{model_type}_{label}"
        model_path = os.path.join(MODEL_DIR, f"opt_{model_type}_{label}.joblib")

        if not is_done(ck, model_key):
            print(f"  Training OPT {model_type} for {label} ({len(y)} samples)...")
            if model_type == "reg":
                train_regressor(X, y, model_path)
            else:
                train_ranker(X, y, g, model_path)
            mark_done(ck, model_key)
        else:
            print(f"  OPT {model_type} {label} already trained.")


def evaluate_models(ck, model_type):
    print(f"\nLoading eval data...")
    reqs, warmup_n = load_eval_data()
    print(f"  {len(reqs)} requests, {warmup_n} warmup")

    for label in TRAIN_CONFIGS:
        n_shards = TRAIN_CONFIGS[label][0]
        max_rows = TRAIN_CONFIGS[label][1]
        name = f"OPT_Imit_{'Reg' if model_type == 'reg' else 'Rank'}_{label}"
        key = f"eval_{name}"
        model_path = os.path.join(MODEL_DIR, f"opt_{model_type}_{label}.joblib")

        if is_done(ck, key):
            print(f"  {name} already evaluated.")
            continue
        if not os.path.exists(model_path):
            print(f"  {name}: model not found, skipping")
            continue

        approx_rows = max_rows if max_rows else n_shards * 142000
        pol = make_ml_policy(FAST_CAP, model_path, fast_fill_threshold=0.9)
        row = run_eval(name, pol, reqs, warmup_n, train_label=label, train_rows=approx_rows)
        append_result(row)
        mark_done(ck, key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["reg", "rank", "both"], default="both")
    args = parser.parse_args()

    ck = load_checkpoint()
    types = ["reg", "rank"] if args.type == "both" else [args.type]

    for mt in types:
        print(f"\n{'='*60}")
        print(f"OPT Imit {'Regressor' if mt == 'reg' else 'Ranker'}")
        print(f"{'='*60}")
        generate_and_train(ck, mt)
        evaluate_models(ck, mt)

    print("\nDone!")
