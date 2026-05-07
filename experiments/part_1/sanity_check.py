"""Quick sanity check: run LRU + a few ML models on small eval to verify pipeline works."""
import os, sys, time, warnings
warnings.filterwarnings("ignore")

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT)

from src.config import EvaluatorConfig, TierConfig
from src.evaluator import Evaluator
from src.trace_loader import load_thesios_csv
from src.policy import make_lru_policy, make_ml_policy

GB, MB = 1024**3, 1024**2
FAST_CAP = 10 * GB
TRACE_DIR = os.path.join(PROJECT, "traces/workload2/cluster_1_group_1")
SHARDS = sorted(os.listdir(TRACE_DIR))
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

SLOW_CFG = TierConfig(200*MB, 150*MB, 5e-3, 5e-3, 8, 8)
FAST_CFG = TierConfig(3.5*GB, 3.0*GB, 20e-6, 20e-6, 32, 32, FAST_CAP)

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

def run_one(name, pol, reqs, warmup_n):
    cfg = EvaluatorConfig(
        FAST_CFG, SLOW_CFG, shard_path(0), "/dev/null",
        warmup_ops=warmup_n, always_write_slow=True, on_fast_full="spill",
    )
    t0 = time.perf_counter()
    mc = Evaluator(cfg, pol, reqs).run()
    elapsed = time.perf_counter() - t0
    s = mc.global_metrics.summary()
    hr = s["hit_rate"]
    print(f"  {name:<30s} hit={hr:.4f}  mean={s['mean_response_time_s']*1000:.3f}ms  ({elapsed:.0f}s)")
    return hr

print("Loading eval data (shards 4-5, 50% warmup)...")
reqs, warmup_n = load_eval_data()
print(f"  {len(reqs)} requests, {warmup_n} warmup\n")

print("=== Sanity Checks ===")
lru_hr = run_one("LRU", make_lru_policy(FAST_CAP), reqs, warmup_n)

models_to_check = [
    ("ML_Reg_10k", "ml_reg_10k.joblib"),
    ("ML_Reg_4shards", "ml_reg_4shards.joblib"),
    ("OPT_Imit_Reg_4shards", "opt_reg_4shards.joblib"),
]

for name, fname in models_to_check:
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        print(f"  {name}: MODEL NOT FOUND at {path}")
        continue
    pol = make_ml_policy(FAST_CAP, path, fast_fill_threshold=0.9)
    hr = run_one(name, pol, reqs, warmup_n)
    if hr < 0.30:
        print(f"  *** WARNING: {name} hit rate {hr:.4f} is below 30% threshold! ***")

print(f"\nLRU baseline: {lru_hr:.4f}")
if lru_hr < 0.60:
    print("*** WARNING: LRU hit rate suspiciously low, check eval setup ***")
else:
    print("LRU looks reasonable.")
print("\nSanity check complete.")
