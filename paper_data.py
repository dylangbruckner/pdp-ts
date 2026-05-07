"""Generate all data for research paper figures.

Outputs CSV files to results/ for plotting by paper_plots.py.
Experiments:
  1. Policy comparison (hit rate, WA, latency) at fixed tier size
  2. Fast tier size scaling (512MB → 20GB)
  3. ML shard distance (train on shard N, eval on shard N+k for k=1..10)
  4. Hit rate over time (temporal degradation of ML vs LRU)
  5. Training data size effect on ML
"""
import csv, os, time, sys
import warnings
warnings.filterwarnings("ignore")

from src.config import EvaluatorConfig, TierConfig
from src.evaluator import Evaluator
from src.trace_loader import load_thesios_csv
from src.policy import (
    make_lru_policy, make_lfu_policy, make_s3fifo_policy,
    make_ml_policy, make_hybrid_policy, make_opt_policy,
)
from src.decay_policy import make_decay_policy

GB, MB = 1024**3, 1024**2
BASE = "traces/workload2/raw/"
SHARDS = sorted([f for f in os.listdir(BASE) if f.startswith("cluster")])
os.makedirs("results", exist_ok=True)

slow_cfg = TierConfig(200*MB, 150*MB, 5e-3, 5e-3, 8, 8)


def load_shards(indices, warmup_frac=0.5):
    reqs = []
    for i in indices:
        reqs.extend(load_thesios_csv(BASE + SHARDS[i]))
    reqs.sort(key=lambda r: r.arrival_time)
    wn = int(len(reqs) * warmup_frac)
    for r in reqs[:wn]:
        r.is_warmup = True
    return reqs, wn


def run_eval(policy, reqs, warmup_n, fast_cap):
    fast = TierConfig(3.5*GB, 3.0*GB, 20e-6, 20e-6, 32, 32, fast_cap)
    cfg = EvaluatorConfig(fast, slow_cfg, BASE + SHARDS[0], "/tmp/paper.csv",
                          warmup_ops=warmup_n, always_write_slow=True,
                          on_fast_full="spill")
    t0 = time.perf_counter()
    ev = Evaluator(cfg, policy, reqs)
    mc = ev.run()
    elapsed = time.perf_counter() - t0
    s = mc.global_metrics.summary()
    wa = mc.global_metrics.write_amplification(mc._requested_write_bytes)
    lat = mc.global_metrics.latency_summary()
    return {
        "hit_rate": s["hit_rate"], "wa": wa,
        "mean_resp_ms": s["mean_response_time_s"] * 1000,
        "p50_resp_ms": s["p50_response_time_s"] * 1000,
        "p99_resp_ms": s["p99_response_time_s"] * 1000,
        "mean_hit_ms": s["mean_response_time_hit_s"] * 1000,
        "mean_miss_ms": s["mean_response_time_miss_s"] * 1000,
        "p99_hit_ms": s["p99_response_time_hit_s"] * 1000,
        "p99_miss_ms": s["p99_response_time_miss_s"] * 1000,
        "mean_fast_ms": s["mean_latency_fast_s"] * 1000,
        "mean_slow_ms": s["mean_latency_slow_s"] * 1000,
        "p99_fast_ms": s["p99_latency_fast_s"] * 1000,
        "p99_slow_ms": s["p99_latency_slow_s"] * 1000,
        "stall_s": s["total_fast_stall_time_s"],
        "bg_acts": s["dpa_background_activations"],
        "wall_s": elapsed,
    }, mc


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {path} ({len(rows)} rows)")


def get_time_series(mc, bucket_s=30.0):
    """Bucket non-warmup requests by arrival time, compute per-bucket hit rate."""
    buckets = {}
    for rec in mc._records:
        if rec.is_warmup:
            continue
        b = int(rec.arrival_time / bucket_s) * bucket_s
        if b not in buckets:
            buckets[b] = {"hits_bytes": 0, "total_bytes": 0, "latencies": []}
        if rec.request_type == "READ":
            buckets[b]["total_bytes"] += rec.request_size
            if rec.hit:
                buckets[b]["hits_bytes"] += rec.request_size
        lat = rec.completion_time - rec.arrival_time
        buckets[b]["latencies"].append(lat)

    import numpy as np
    result = []
    for t in sorted(buckets):
        b = buckets[t]
        hr = b["hits_bytes"] / max(1, b["total_bytes"])
        lats = b["latencies"]
        result.append({
            "time_s": t,
            "hit_rate": round(hr, 4),
            "mean_resp_ms": round(float(np.mean(lats)) * 1000, 3),
            "p99_resp_ms": round(float(np.percentile(lats, 99)) * 1000, 3) if lats else 0,
        })
    return result


# ═══════════════════════════════════════════════════════════════
# Experiment 1: Policy comparison at 512 MB (eval shards 2-3)
# ═══════════════════════════════════════════════════════════════
def exp1_policy_comparison():
    print("\n=== EXP 1: Policy comparison (512 MB fast, shards 2-3) ===")
    reqs, wn = load_shards([2, 3])
    cap = 512 * MB
    rows = []
    fields = ["policy", "hit_rate", "wa", "mean_resp_ms", "p50_resp_ms", "p99_resp_ms",
              "mean_hit_ms", "mean_miss_ms", "p99_hit_ms", "p99_miss_ms",
              "mean_fast_ms", "mean_slow_ms", "p99_fast_ms", "p99_slow_ms",
              "stall_s", "bg_acts", "wall_s"]

    policies = [
        ("LRU", make_lru_policy(cap)),
        ("LFU", make_lfu_policy(cap, promotion_threshold=0.5)),
        ("S3FIFO", make_s3fifo_policy(cap)),
        ("Decay (best)", make_decay_policy(cap, alpha=0.5, decay_rate=0.05, fast_fill_threshold=0.9)),
    ]
    if os.path.exists("models/coldness_v2.joblib"):
        policies.append(("ML Regressor", make_ml_policy(cap, "models/coldness_v2.joblib", fast_fill_threshold=0.9)))
    if os.path.exists("models/coldness_v3_ranker.joblib"):
        policies.append(("ML Ranker", make_ml_policy(cap, "models/coldness_v3_ranker.joblib", fast_fill_threshold=0.9)))
    if os.path.exists("models/coldness_v2.joblib"):
        policies.append(("Hybrid p=10%", make_hybrid_policy(cap, "models/coldness_v2.joblib", fast_fill_threshold=0.9, protect_percentile=0.10)))
    policies.append(("OPT (oracle)", make_opt_policy(cap, reqs, fast_fill_threshold=0.9, background_interval=5.0)))

    for name, pol in policies:
        print(f"  Running {name}...", end="", flush=True)
        r, _ = run_eval(pol, reqs, wn, cap)
        r["policy"] = name
        rows.append(r)
        print(f" hit={r['hit_rate']:.4f}")

    write_csv("results/exp1_policy_comparison.csv", rows, fields)


# ═══════════════════════════════════════════════════════════════
# Experiment 2: Fast tier size scaling
# ═══════════════════════════════════════════════════════════════
def exp2_tier_scaling():
    print("\n=== EXP 2: Tier size scaling (shards 2-3) ===")
    reqs, wn = load_shards([2, 3])
    rows = []
    fields = ["fast_cap_mb", "policy", "hit_rate", "wa", "mean_resp_ms", "p99_resp_ms"]

    for cap_mb in [512, 1024, 2048, 5120, 10240, 20480]:
        cap = cap_mb * MB
        for name, make_fn in [
            ("LRU", lambda c: make_lru_policy(c)),
            ("Decay", lambda c: make_decay_policy(c, alpha=0.5, decay_rate=0.05, fast_fill_threshold=0.9)),
        ]:
            print(f"  {name} @ {cap_mb} MB...", end="", flush=True)
            pol = make_fn(cap)
            r, _ = run_eval(pol, reqs, wn, cap)
            rows.append({"fast_cap_mb": cap_mb, "policy": name,
                          "hit_rate": r["hit_rate"], "wa": r["wa"],
                          "mean_resp_ms": r["mean_resp_ms"], "p99_resp_ms": r["p99_resp_ms"]})
            print(f" hit={r['hit_rate']:.4f}")

    write_csv("results/exp2_tier_scaling.csv", rows, fields)


# ═══════════════════════════════════════════════════════════════
# Experiment 3: ML shard distance (generalization degradation)
# ═══════════════════════════════════════════════════════════════
def exp3_shard_distance():
    print("\n=== EXP 3: ML shard distance ===")
    cap = 512 * MB
    rows = []
    fields = ["train_shard", "eval_shard", "distance", "policy", "hit_rate"]

    train_shard = 0
    model_path = "models/coldness_v2.joblib"
    if not os.path.exists(model_path):
        print("  SKIP: no model")
        return

    for eval_idx in [1, 2, 4, 6, 8, 10, 15, 20]:
        if eval_idx >= len(SHARDS):
            break
        print(f"  Eval shard {eval_idx} (distance={eval_idx})...", end="", flush=True)
        reqs, wn = load_shards([eval_idx])

        lru_r, _ = run_eval(make_lru_policy(cap), reqs, wn, cap)
        ml_r, _ = run_eval(make_ml_policy(cap, model_path, fast_fill_threshold=0.9), reqs, wn, cap)
        decay_r, _ = run_eval(make_decay_policy(cap, alpha=0.5, decay_rate=0.05, fast_fill_threshold=0.9), reqs, wn, cap)

        for name, r in [("LRU", lru_r), ("ML", ml_r), ("Decay", decay_r)]:
            rows.append({"train_shard": train_shard, "eval_shard": eval_idx,
                          "distance": eval_idx, "policy": name, "hit_rate": r["hit_rate"]})
        print(f" LRU={lru_r['hit_rate']:.4f} ML={ml_r['hit_rate']:.4f} Decay={decay_r['hit_rate']:.4f}")

    write_csv("results/exp3_shard_distance.csv", rows, fields)


# ═══════════════════════════════════════════════════════════════
# Experiment 4: Hit rate over time (temporal degradation)
# ═══════════════════════════════════════════════════════════════
def exp4_temporal():
    print("\n=== EXP 4: Hit rate over time (shard 2) ===")
    reqs, wn = load_shards([2])
    cap = 512 * MB
    all_ts = []

    model_path = "models/coldness_v2.joblib"
    policies = [("LRU", make_lru_policy(cap))]
    if os.path.exists(model_path):
        policies.append(("ML", make_ml_policy(cap, model_path, fast_fill_threshold=0.9)))
    policies.append(("Decay", make_decay_policy(cap, alpha=0.5, decay_rate=0.05, fast_fill_threshold=0.9)))
    policies.append(("OPT", make_opt_policy(cap, reqs, fast_fill_threshold=0.9, background_interval=5.0)))

    for name, pol in policies:
        print(f"  {name}...", end="", flush=True)
        _, mc = run_eval(pol, reqs, wn, cap)
        ts = get_time_series(mc, bucket_s=10.0)
        for row in ts:
            row["policy"] = name
        all_ts.extend(ts)
        print(f" {len(ts)} buckets")

    fields = ["policy", "time_s", "hit_rate", "mean_resp_ms", "p99_resp_ms"]
    write_csv("results/exp4_temporal.csv", all_ts, fields)


# ═══════════════════════════════════════════════════════════════
# Experiment 5: Training data size effect
# ═══════════════════════════════════════════════════════════════
def exp5_training_size():
    print("\n=== EXP 5: Training data size effect ===")
    from src.ml_training import generate_training_data, train_model
    import numpy as np

    cap = 512 * MB
    eval_reqs, eval_wn = load_shards([2])
    rows = []
    fields = ["train_rows", "policy", "hit_rate", "wa"]

    for max_rows in [10000, 50000, 100000, 200000]:
        train_reqs = load_thesios_csv(BASE + SHARDS[0], max_rows=max_rows, warmup_rows=0)
        print(f"  Training on {len(train_reqs)} rows...", end="", flush=True)

        X, y = generate_training_data(train_reqs, fast_capacity=cap)
        path = f"models/paper_train_{max_rows}.joblib"
        train_model(X, y, path)

        ml_r, _ = run_eval(make_ml_policy(cap, path, fast_fill_threshold=0.9),
                           eval_reqs, eval_wn, cap)
        rows.append({"train_rows": max_rows, "policy": "ML", "hit_rate": ml_r["hit_rate"], "wa": ml_r["wa"]})
        print(f" hit={ml_r['hit_rate']:.4f}")

    lru_r, _ = run_eval(make_lru_policy(cap), eval_reqs, eval_wn, cap)
    for n in [10000, 50000, 100000, 200000]:
        rows.append({"train_rows": n, "policy": "LRU", "hit_rate": lru_r["hit_rate"], "wa": lru_r["wa"]})

    write_csv("results/exp5_training_size.csv", rows, fields)


# ═══════════════════════════════════════════════════════════════
# Run selected experiments
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    experiments = {
        "1": exp1_policy_comparison,
        "2": exp2_tier_scaling,
        "3": exp3_shard_distance,
        "4": exp4_temporal,
        "5": exp5_training_size,
    }

    to_run = sys.argv[1:] if len(sys.argv) > 1 else ["1", "2", "3", "4", "5"]

    for exp_id in to_run:
        if exp_id in experiments:
            experiments[exp_id]()
        else:
            print(f"Unknown experiment: {exp_id}")

    print("\nAll done. CSV files in results/")
