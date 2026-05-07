"""Part 8: Adaptive ML policy — LRU with ML tie-breaking.

Evaluates the adaptive ML policy at 10GB with shards 4-5, 50% warmup.
Sweeps tie_window and ml_weight parameters. Compares against LRU and
best pure-ML (OPT Imit Rank).

Uses OPT_Rank_t100_d4_l15 model from Part 2 (best pure ML model).
"""
import csv, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PART_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)

from src.config import EvaluatorConfig, TierConfig
from src.evaluator import Evaluator
from src.trace_loader import load_thesios_csv
from src.policy import make_lru_policy, make_hybrid_policy
from src.adaptive_ml_policy import make_adaptive_ml_policy
from src.decay_policy import make_decay_policy
from src.ml_policy import make_ml_policy

GB, MB = 1024**3, 1024**2
FAST_CAP = 10 * GB
TRACE_DIR = os.path.join(PROJECT, "traces/workload2/cluster_1_group_1")
SHARDS = sorted(os.listdir(TRACE_DIR))

MODEL_PATH = os.path.join(PROJECT, "experiments/part_2/models/OPT_Rank_t100_d4_l15.joblib")
MODEL_OPT_RANK_10G = os.path.join(PROJECT, "experiments/part_5/models/opt_rank_10GB.joblib")
MODEL_ML_REG_10G = os.path.join(PROJECT, "experiments/part_5/models/ml_reg_10GB.joblib")

RESULT_DIR = os.path.join(PART_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

SLOW_CFG = TierConfig(200*MB, 150*MB, 5e-3, 5e-3, 8, 8)
FAST_CFG = TierConfig(3.5*GB, 3.0*GB, 20e-6, 20e-6, 32, 32, FAST_CAP)

CSV_FIELDS = ["policy", "model", "tie_window", "ml_weight", "protect_recent",
              "hit_rate", "wa", "mean_resp_ms", "p99_resp_ms", "wall_s"]
RESULTS_CSV = os.path.join(RESULT_DIR, "part8_results.csv")

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

def run_eval(name, pol, reqs, warmup_n, extra=None):
    cfg = EvaluatorConfig(
        FAST_CFG, SLOW_CFG, shard_path(0), "/dev/null",
        warmup_ops=warmup_n, always_write_slow=True, on_fast_full="spill",
    )
    t0 = time.perf_counter()
    mc = Evaluator(cfg, pol, reqs).run()
    elapsed = time.perf_counter() - t0
    s = mc.global_metrics.summary()
    wa = mc.global_metrics.write_amplification(mc._requested_write_bytes)
    hit = s["hit_rate"]
    mean_ms = s["mean_response_time_s"] * 1000
    p99_ms = s["p99_response_time_s"] * 1000
    row = {
        "policy": name,
        "model": extra.get("model", "") if extra else "",
        "tie_window": extra.get("tie_window", "") if extra else "",
        "ml_weight": extra.get("ml_weight", "") if extra else "",
        "protect_recent": extra.get("protect_recent", "") if extra else "",
        "hit_rate": f"{hit:.4f}",
        "wa": f"{wa:.3f}",
        "mean_resp_ms": f"{mean_ms:.3f}",
        "p99_resp_ms": f"{p99_ms:.3f}",
        "wall_s": f"{elapsed:.1f}",
    }
    print(f"  {name}: hit={hit:.4f} mean={mean_ms:.3f}ms "
          f"p99={p99_ms:.3f}ms  ({elapsed:.0f}s)")
    append_result(row)
    return s

def append_result(row):
    exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)

def load_checkpoint():
    ck_path = os.path.join(RESULT_DIR, "checkpoint.json")
    if os.path.exists(ck_path):
        with open(ck_path) as f:
            return json.load(f)
    return {"completed": []}

def save_checkpoint(ck):
    with open(os.path.join(RESULT_DIR, "checkpoint.json"), "w") as f:
        json.dump(ck, f, indent=2)

def main():
    print("Loading eval data (shards 4-5, 50% warmup)...")
    reqs, warmup_n = load_eval_data()
    print(f"  {len(reqs)} requests, {warmup_n} warmup")

    ck = load_checkpoint()

    # Pick best available model
    if os.path.exists(MODEL_OPT_RANK_10G):
        model_path = MODEL_OPT_RANK_10G
        model_name = "opt_rank_10GB"
    elif os.path.exists(MODEL_PATH):
        model_path = MODEL_PATH
        model_name = "OPT_Rank_t100_d4"
    else:
        print("ERROR: No model found!")
        return

    print(f"Using model: {model_name} ({model_path})")

    # --- Baselines ---
    key = "LRU"
    if key not in ck["completed"]:
        print(f"\n[Baseline] {key}")
        pol = make_lru_policy(FAST_CAP)
        run_eval(key, pol, reqs, warmup_n)
        ck["completed"].append(key)
        save_checkpoint(ck)

    key = "Decay"
    if key not in ck["completed"]:
        print(f"\n[Baseline] {key}")
        pol = make_decay_policy(FAST_CAP, alpha=0.5, decay_rate=0.05)
        run_eval(key, pol, reqs, warmup_n)
        ck["completed"].append(key)
        save_checkpoint(ck)

    key = "OPT_Imit_Rank_pure"
    if key not in ck["completed"]:
        print(f"\n[Baseline] {key}")
        pol = make_ml_policy(FAST_CAP, model_path)
        run_eval(key, pol, reqs, warmup_n, {"model": model_name})
        ck["completed"].append(key)
        save_checkpoint(ck)

    key = "Hybrid_OPT_Rank_p10"
    if key not in ck["completed"]:
        print(f"\n[Baseline] {key}")
        pol = make_hybrid_policy(FAST_CAP, model_path, protect_percentile=0.10)
        run_eval(key, pol, reqs, warmup_n, {"model": model_name})
        ck["completed"].append(key)
        save_checkpoint(ck)

    # --- Adaptive ML: sweep tie_window ---
    print("\n--- Sweep: tie_window (ml_weight=0.5, protect=2.0s) ---")
    for tw in [2.0, 5.0, 10.0, 20.0, 60.0]:
        key = f"Adaptive_tw{tw}_mw0.5_pr2.0"
        if key in ck["completed"]:
            continue
        print(f"\n  {key}")
        pol = make_adaptive_ml_policy(
            FAST_CAP, model_path,
            tie_window=tw, ml_weight=0.5, protect_recent_seconds=2.0,
        )
        run_eval(key, pol, reqs, warmup_n, {
            "model": model_name, "tie_window": tw,
            "ml_weight": 0.5, "protect_recent": 2.0,
        })
        ck["completed"].append(key)
        save_checkpoint(ck)

    # --- Sweep ml_weight with tw=10 ---
    print("\n--- Sweep: ml_weight (tie_window=10.0, protect=2.0s) ---")
    for mw in [0.0, 0.2, 0.5, 0.8, 1.0]:
        key = f"Adaptive_tw10.0_mw{mw}_pr2.0"
        if key in ck["completed"]:
            continue
        print(f"\n  {key}")
        pol = make_adaptive_ml_policy(
            FAST_CAP, model_path,
            tie_window=10.0, ml_weight=mw, protect_recent_seconds=2.0,
        )
        run_eval(key, pol, reqs, warmup_n, {
            "model": model_name, "tie_window": 10.0,
            "ml_weight": mw, "protect_recent": 2.0,
        })
        ck["completed"].append(key)
        save_checkpoint(ck)

    # --- Sweep protect_recent ---
    print("\n--- Sweep: protect_recent (tie_window=10.0, ml_weight=0.5) ---")
    for pr in [0.0, 1.0, 2.0, 5.0, 10.0]:
        key = f"Adaptive_tw10.0_mw0.5_pr{pr}"
        if key in ck["completed"]:
            continue
        print(f"\n  {key}")
        pol = make_adaptive_ml_policy(
            FAST_CAP, model_path,
            tie_window=10.0, ml_weight=0.5, protect_recent_seconds=pr,
        )
        run_eval(key, pol, reqs, warmup_n, {
            "model": model_name, "tie_window": 10.0,
            "ml_weight": 0.5, "protect_recent": pr,
        })
        ck["completed"].append(key)
        save_checkpoint(ck)

    # --- Try with ML Reg model ---
    if os.path.exists(MODEL_ML_REG_10G):
        print("\n--- Adaptive with ML Reg model ---")
        key = "Adaptive_MLReg_tw10.0_mw0.5_pr2.0"
        if key not in ck["completed"]:
            pol = make_adaptive_ml_policy(
                FAST_CAP, MODEL_ML_REG_10G,
                tie_window=10.0, ml_weight=0.5, protect_recent_seconds=2.0,
            )
            run_eval(key, pol, reqs, warmup_n, {
                "model": "ml_reg_10GB", "tie_window": 10.0,
                "ml_weight": 0.5, "protect_recent": 2.0,
            })
            ck["completed"].append(key)
            save_checkpoint(ck)

    # --- Extreme configs for understanding ---
    # Pure ML reorder (tie_window=inf)
    key = "Adaptive_tw9999_mw1.0_pr0.0"
    if key not in ck["completed"]:
        print("\n--- Pure ML reorder (tie_window=inf) ---")
        pol = make_adaptive_ml_policy(
            FAST_CAP, model_path,
            tie_window=9999.0, ml_weight=1.0, protect_recent_seconds=0.0,
        )
        run_eval(key, pol, reqs, warmup_n, {
            "model": model_name, "tie_window": 9999.0,
            "ml_weight": 1.0, "protect_recent": 0.0,
        })
        ck["completed"].append(key)
        save_checkpoint(ck)

    # Pure LRU sanity check (tie_window=0)
    key = "Adaptive_tw0.0_mw0.5_pr0.0"
    if key not in ck["completed"]:
        print("\n--- Pure LRU sanity check (tie_window=0) ---")
        pol = make_adaptive_ml_policy(
            FAST_CAP, model_path,
            tie_window=0.0, ml_weight=0.5, protect_recent_seconds=0.0,
        )
        run_eval(key, pol, reqs, warmup_n, {
            "model": model_name, "tie_window": 0.0,
            "ml_weight": 0.5, "protect_recent": 0.0,
        })
        ck["completed"].append(key)
        save_checkpoint(ck)

    print(f"\n\nDone! Results in: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
