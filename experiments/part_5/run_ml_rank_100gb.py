"""Run just ML_Rank eval at 100GB with 2 eval shards (4-5)."""
import os, sys, time, warnings
warnings.filterwarnings("ignore")

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT)

from src.config import EvaluatorConfig, TierConfig
from src.evaluator import Evaluator
from src.trace_loader import load_thesios_csv
from src.policy import make_ml_policy

GB, MB = 1024**3, 1024**2
TRACE_DIR = os.path.join(PROJECT, "traces/workload2/cluster_1_group_1")
SHARDS = sorted(os.listdir(TRACE_DIR))
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

SLOW_CFG = TierConfig(200*MB, 150*MB, 5e-3, 5e-3, 8, 8)
cap = 100 * GB

def shard_path(idx):
    return os.path.join(TRACE_DIR, SHARDS[idx])

reqs = []
for i in range(4, 6):
    reqs.extend(load_thesios_csv(shard_path(i)))
reqs.sort(key=lambda r: r.arrival_time)
warmup_n = len(reqs) // 2
for r in reqs[:warmup_n]:
    r.is_warmup = True

print(f"Eval: {len(reqs)} requests, {warmup_n} warmup")

model_path = os.path.join(MODEL_DIR, "ml_rank_100GB.joblib")
pol = make_ml_policy(cap, model_path, fast_fill_threshold=0.9)

fast_cfg = TierConfig(3.5*GB, 3.0*GB, 20e-6, 20e-6, 32, 32, cap)
cfg = EvaluatorConfig(
    fast_cfg, SLOW_CFG, shard_path(0), "/dev/null",
    warmup_ops=warmup_n, always_write_slow=True, on_fast_full="spill",
)

t0 = time.perf_counter()
mc = Evaluator(cfg, pol, reqs).run()
elapsed = time.perf_counter() - t0

s = mc.global_metrics.summary()
wa = mc.global_metrics.write_amplification(mc._requested_write_bytes)
print(f"ML_Rank @100GB hit={s['hit_rate']:.4f} wa={wa:.4f} mean={s['mean_response_time_s']*1000:.3f}ms p99={s['p99_response_time_s']*1000:.3f}ms ({elapsed:.0f}s)")
