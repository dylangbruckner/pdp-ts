"""Train robust models on shards 0-1, evaluate on shards 2-3."""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import joblib
import numpy as np

from src.robust_features import RobustTracker, NUM_ROBUST_FEATURES
from src.trace_loader import load_thesios_csv
from src.models import OpType

BASE = "traces/workload2/raw/"
MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)

shards = sorted([f for f in os.listdir(BASE) if f.startswith("cluster")])
print(f"Found {len(shards)} shards: {shards}")


# --- STEP 1: Generate training data from shards 0-1 ---
def generate_robust_training_data(
    requests: List,
    fast_capacity: float = 512 * 1024**2,
    max_target: float = 600.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sorted_reqs = sorted(requests, key=lambda r: r.arrival_time)

    next_access = [float("inf")] * len(sorted_reqs)
    last_seen: Dict[str, float] = {}
    for i in range(len(sorted_reqs) - 1, -1, -1):
        fid = sorted_reqs[i].file_id
        if fid in last_seen:
            next_access[i] = last_seen[fid]
        last_seen[fid] = sorted_reqs[i].arrival_time

    tracker = RobustTracker()
    X_list, y_list, t_list = [], [], []
    tracked_size = 0.0

    for i, req in enumerate(sorted_reqs):
        fid = req.file_id
        in_tracker = fid in tracker.tracked_files

        if in_tracker:
            vec = tracker.feature_vector(fid, req.arrival_time)
            time_to_next = min(next_access[i] - req.arrival_time, max_target)
            X_list.append(vec)
            y_list.append(time_to_next)
            t_list.append(req.arrival_time)

        tracker.update(fid, req.arrival_time, req.op_type, req.size, req.offset)

        if not in_tracker:
            if tracked_size < fast_capacity or req.op_type == OpType.WRITE:
                tracker.add(fid, req.arrival_time, req.size)
                tracked_size += req.size

    X = np.array(X_list) if X_list else np.empty((0, NUM_ROBUST_FEATURES))
    y = np.array(y_list) if y_list else np.empty(0)
    ts = np.array(t_list) if t_list else np.empty(0)
    return X, y, ts


print("\n=== Loading shards 0-1 for training ===")
train_reqs = []
for s in [shards[0], shards[1]]:
    print(f"  Loading {s}")
    train_reqs.extend(load_thesios_csv(os.path.join(BASE, s)))
train_reqs.sort(key=lambda r: r.arrival_time)
print(f"  Total training requests: {len(train_reqs)}")

print("\nGenerating training data...")
X_train, y_train, ts_train = generate_robust_training_data(train_reqs)
print(f"  Samples: {X_train.shape[0]}, target mean: {y_train.mean():.2f}s, median: {np.median(y_train):.2f}s")
print(f"  Features: {RobustTracker.feature_names()}")


# --- STEP 2: Train models ---
def make_relevance_labels(y):
    labels = np.zeros(len(y), dtype=np.int32)
    labels[y > 1] = 1
    labels[y > 10] = 2
    labels[y > 60] = 3
    labels[y > 300] = 4
    return labels


def make_groups(timestamps, window=10.0):
    group_ids = (timestamps // window).astype(np.int64)
    _, counts = np.unique(group_ids, return_counts=True)
    return counts


import lightgbm as lgb

# Regressor
print("\n=== Training robust regressor ===")
y_log = np.log1p(y_train)
reg = lgb.LGBMRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    num_leaves=31, min_child_samples=50, verbose=-1,
)
reg.fit(X_train, y_log)
reg_path = os.path.join(MODEL_DIR, "robust_regressor.joblib")
joblib.dump(reg, reg_path)
print(f"  Saved to {reg_path}")

# Ranker
print("\n=== Training robust ranker ===")
relevance = make_relevance_labels(y_train)
group_sizes = make_groups(ts_train, window=10.0)
assert group_sizes.sum() == len(y_train)

ranker = lgb.LGBMRanker(
    objective="lambdarank", n_estimators=500, max_depth=6,
    learning_rate=0.05, num_leaves=31, min_child_samples=50,
    verbose=-1, label_gain=[0, 1, 3, 7, 15],
)
ranker.fit(X_train, relevance, group=group_sizes)
ranker_path = os.path.join(MODEL_DIR, "robust_ranker.joblib")
joblib.dump(ranker, ranker_path)
print(f"  Saved to {ranker_path}")
print(f"  Relevance distribution: {np.bincount(relevance)}")


# --- STEP 3: Evaluate on shards 2-3 ---
print("\n=== Loading shards 2-3 for evaluation ===")
eval_reqs = []
for s in [shards[2], shards[3]]:
    print(f"  Loading {s}")
    eval_reqs.extend(load_thesios_csv(os.path.join(BASE, s)))
eval_reqs.sort(key=lambda r: r.arrival_time)
warmup_n = len(eval_reqs) // 2
for r in eval_reqs[:warmup_n]:
    r.is_warmup = True
print(f"  Total eval requests: {len(eval_reqs)}, warmup: {warmup_n}")

from src.config import EvaluatorConfig, TierConfig
from src.evaluator import Evaluator
from src.policy import make_lru_policy
from src.robust_ml_policy import make_robust_ml_policy, make_robust_hybrid_policy

GB, MB = 1024**3, 1024**2
FAST_CAP = 512 * MB
fast = TierConfig(3.5*GB, 3.0*GB, 20e-6, 20e-6, 32, 32, FAST_CAP)
slow = TierConfig(200*MB, 150*MB, 5e-3, 5e-3, 8, 8)


def run_eval(name, policy):
    cfg = EvaluatorConfig(fast, slow, os.path.join(BASE, shards[2]), f"/tmp/robust_{name}.csv",
                          warmup_ops=warmup_n, always_write_slow=True, on_fast_full="spill")
    ev = Evaluator(cfg, policy, eval_reqs)
    t0 = time.time()
    mc = ev.run()
    elapsed = time.time() - t0
    s = mc.global_metrics.summary()
    wa = mc.global_metrics.write_amplification(mc._requested_write_bytes)
    print(f"  {name:30s}  hit={s['hit_rate']:.4f}  WA={wa:.3f}  "
          f"stall={s['total_fast_stall_time_s']:.1f}s  time={elapsed:.1f}s")
    return s['hit_rate']


print("\n=== Cross-shard evaluation (train 0-1, eval 2-3) ===")

lru_hit = run_eval("LRU", make_lru_policy(FAST_CAP))

reg_hit = run_eval("Robust-Regressor",
    make_robust_ml_policy(FAST_CAP, reg_path, fast_fill_threshold=0.9))

rank_hit = run_eval("Robust-Ranker",
    make_robust_ml_policy(FAST_CAP, ranker_path, fast_fill_threshold=0.9))

for pct in [0.10, 0.25]:
    run_eval(f"Robust-Hybrid-Reg-p{int(pct*100)}",
        make_robust_hybrid_policy(FAST_CAP, reg_path,
                                  protect_percentile=pct, fast_fill_threshold=0.9))

for pct in [0.10, 0.25]:
    run_eval(f"Robust-Hybrid-Rank-p{int(pct*100)}",
        make_robust_hybrid_policy(FAST_CAP, ranker_path,
                                  protect_percentile=pct, fast_fill_threshold=0.9))

print("\n=== Summary ===")
target = 0.7375
print(f"LRU baseline: {lru_hit:.4f}")
print(f"Target (within 10% of LRU): {target:.4f}")
print(f"Robust Regressor: {reg_hit:.4f} ({'PASS' if reg_hit >= target else 'FAIL'})")
print(f"Robust Ranker: {rank_hit:.4f} ({'PASS' if rank_hit >= target else 'FAIL'})")
