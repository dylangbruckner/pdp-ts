"""Reproduce old OPT Imit Reg (10G) result: eval on shards 2-3."""
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

SLOW_CFG = TierConfig(200*MB, 150*MB, 5e-3, 5e-3, 8, 8)
FAST_CFG = TierConfig(3.5*GB, 3.0*GB, 20e-6, 20e-6, 32, 32, FAST_CAP)

def shard_path(idx):
    return os.path.join(TRACE_DIR, SHARDS[idx])

def load_eval(start, end):
    reqs = []
    for i in range(start, end):
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
    print(f"  {name:<40s} hit={s['hit_rate']:.4f}  mean={s['mean_response_time_s']*1000:.3f}ms  ({elapsed:.0f}s)")

models = [
    ("OLD opt_imit_10g_reg (shards 2-3)", os.path.join(PROJECT, "models/opt_imit_10g_reg.joblib")),
    ("Part1 opt_reg_4shards (shards 2-3)", os.path.join(PROJECT, "experiments/part_1/models/opt_reg_4shards.joblib")),
]

print("Eval on shards 2-3 (50% warmup) — same as old experiment")
reqs, warmup_n = load_eval(2, 4)
print(f"  {len(reqs)} requests, {warmup_n} warmup\n")

lru_pol = make_lru_policy(FAST_CAP)
run_one("LRU baseline", lru_pol, reqs, warmup_n)

for name, path in models:
    if not os.path.exists(path):
        print(f"  {name}: NOT FOUND at {path}")
        continue
    pol = make_ml_policy(FAST_CAP, path, fast_fill_threshold=0.9)
    run_one(name, pol, reqs, warmup_n)
