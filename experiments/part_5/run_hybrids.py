"""Part 5 supplement: run best 2 hybrids across all cache sizes."""
import csv, json, os, sys, time, warnings
warnings.filterwarnings("ignore")

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PART_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT)

from src.config import EvaluatorConfig, TierConfig
from src.evaluator import Evaluator
from src.trace_loader import load_thesios_csv
from src.policy import make_hybrid_policy

GB, MB = 1024**3, 1024**2
TRACE_DIR = os.path.join(PROJECT, "traces/workload2/cluster_1_group_1")
SHARDS = sorted(os.listdir(TRACE_DIR))
MODEL_DIR = os.path.join(PART_DIR, "models")
RESULT_DIR = os.path.join(PART_DIR, "results")

SLOW_CFG = TierConfig(200*MB, 150*MB, 5e-3, 5e-3, 8, 8)

CSV_FIELDS = ["policy", "cache_size_label", "cache_bytes", "hit_rate", "wa",
              "mean_resp_ms", "p99_resp_ms", "wall_s"]
RESULTS_CSV = os.path.join(RESULT_DIR, "part5_hybrids.csv")
CHECKPOINT = os.path.join(RESULT_DIR, "checkpoint_hybrids.json")

CACHE_SIZES = [
    ("512MB", 512*MB), ("1GB", 1*GB), ("2GB", 2*GB), ("5GB", 5*GB),
    ("10GB", 10*GB), ("20GB", 20*GB), ("100GB", 100*GB),
]

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

def make_fast_cfg(cap):
    return TierConfig(3.5*GB, 3.0*GB, 20e-6, 20e-6, 32, 32, cap)

def run_eval(name, pol, reqs, warmup_n, cap, size_label):
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
        "hit_rate": round(s["hit_rate"], 6), "wa": round(wa, 4),
        "mean_resp_ms": round(s["mean_response_time_s"] * 1000, 4),
        "p99_resp_ms": round(s["p99_response_time_s"] * 1000, 4),
        "wall_s": round(elapsed, 1),
    }
    print(f"  {name:<30s} @{size_label:<6s} hit={row['hit_rate']:.4f}  ({elapsed:.0f}s)")
    return row


if __name__ == "__main__":
    ck = load_checkpoint()
    reqs, warmup_n = load_eval_data()
    print(f"Eval: {len(reqs)} requests, {warmup_n} warmup\n")

    for size_label, cap in CACHE_SIZES:
        ml_reg_path = os.path.join(MODEL_DIR, f"ml_reg_{size_label}.joblib")
        opt_rank_path = os.path.join(MODEL_DIR, f"opt_rank_{size_label}.joblib")

        hybrids = [
            (f"Hybrid_ML_Reg_p25", ml_reg_path, 0.25),
            (f"Hybrid_OPT_Rank_p10", opt_rank_path, 0.10),
        ]

        for name, model_path, pct in hybrids:
            key = f"{name}_{size_label}"
            if is_done(ck, key):
                print(f"  {name} @{size_label} already done.")
                continue
            if not os.path.exists(model_path):
                print(f"  {name} @{size_label}: model not found at {model_path}, skipping")
                continue
            pol = make_hybrid_policy(cap, model_path,
                                     protect_percentile=pct, fast_fill_threshold=0.9)
            row = run_eval(name, pol, reqs, warmup_n, cap, size_label)
            append_result(row)
            mark_done(ck, key)

    print("\nDone!")
