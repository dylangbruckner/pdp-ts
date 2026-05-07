"""Generate imitation learning data from OPT eviction decisions, then train models."""
from __future__ import annotations

import argparse
import bisect
import math
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import joblib

from .ml_features import FileTracker, NUM_FEATURES
from .models import OpType
from .trace_loader import load_thesios_csv

MB = 1024 ** 2


def build_oracle(requests) -> Dict[str, List[float]]:
    future: Dict[str, List[float]] = defaultdict(list)
    for r in sorted(requests, key=lambda r: r.arrival_time):
        future[r.file_id].append(r.arrival_time)
    return dict(future)


def next_access_after(future, fid, t):
    times = future.get(fid)
    if times is None:
        return float("inf")
    idx = bisect.bisect_right(times, t)
    return times[idx] if idx < len(times) else float("inf")


def generate_imitation_data(
    requests,
    fast_capacity: int = 512 * MB,
    bg_interval: float = 5.0,
    fill_threshold: float = 0.9,
    max_candidates_per_group: int = 500,
    sample_rate: int = 1,
    future_requests=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate OPT decisions, collect (features, next_access_time, group_id) at eviction points.

    max_candidates_per_group: cap files sampled per eviction event (controls memory).
    sample_rate: only record every Nth eviction event.
    future_requests: if provided, oracle sees these for future lookups (can extend
      beyond the simulation window so files don't all appear to have no future access).
    """

    future = build_oracle(future_requests if future_requests is not None else requests)
    sorted_reqs = sorted(requests, key=lambda r: r.arrival_time)

    tracker = FileTracker()
    fast_tier: Dict[str, int] = {}  # fid -> size
    fast_used = 0
    lru: OrderedDict = OrderedDict()

    X_all, y_all, g_all = [], [], []
    group_id = 0
    last_bg_time = 0.0
    _eviction_counter = [0]

    def _touch(fid):
        lru[fid] = True
        lru.move_to_end(fid)

    def _collect_eviction_samples(current_time, evict_fid):
        nonlocal group_id
        _eviction_counter[0] += 1
        if _eviction_counter[0] % sample_rate != 0:
            return
        fids_in_tier = list(fast_tier.keys())
        if len(fids_in_tier) < 2:
            return
        # Cap candidates: coldest LRU + evicted file + random sample
        if len(fids_in_tier) > max_candidates_per_group:
            lru_cold = [f for f in lru if f in fast_tier][:max_candidates_per_group // 2]
            remaining = [f for f in fids_in_tier if f not in set(lru_cold)]
            import random
            sample_n = max_candidates_per_group - len(lru_cold)
            sampled = random.sample(remaining, min(sample_n, len(remaining)))
            fids_in_tier = list(set(lru_cold + sampled))
            if evict_fid not in fids_in_tier:
                fids_in_tier.append(evict_fid)
        features = []
        targets = []
        for fid in fids_in_tier:
            fv = tracker.feature_vector(fid, current_time)
            na = next_access_after(future, fid, current_time)
            features.append(fv)
            gap = (na - current_time) if na < float("inf") else 1e8
            targets.append(math.log1p(gap))
        X_all.append(np.array(features))
        y_all.append(np.array(targets))
        g_all.append(np.full(len(fids_in_tier), group_id))
        group_id += 1

    def _opt_pick_evict(current_time, exclude_fid=None):
        best_fid = None
        best_next = -1.0
        no_future = []
        for fid in fast_tier:
            if fid == exclude_fid:
                continue
            na = next_access_after(future, fid, current_time)
            if na == float("inf"):
                no_future.append(fid)
            elif na > best_next:
                best_next = na
                best_fid = fid
        if no_future:
            lru_order = [f for f in lru if f in set(no_future)]
            return lru_order[0] if lru_order else no_future[0]
        return best_fid

    def _do_eviction(fid, current_time):
        nonlocal fast_used
        _collect_eviction_samples(current_time, fid)
        sz = fast_tier.pop(fid)
        fast_used -= sz
        tracker.remove(fid)
        lru.pop(fid, None)

    def _bg_evict(current_time):
        nonlocal fast_used
        if fast_used <= fill_threshold * fast_capacity:
            return
        while fast_used > fill_threshold * fast_capacity and fast_tier:
            victim = _opt_pick_evict(current_time)
            if victim is None:
                break
            _do_eviction(victim, current_time)

    logical_sizes: Dict[str, int] = {}  # fid -> max(offset + io_size)

    for req in sorted_reqs:
        t = req.arrival_time

        if t - last_bg_time >= bg_interval:
            _bg_evict(t)
            last_bg_time = t

        fid = req.file_id
        logical_sizes[fid] = max(logical_sizes.get(fid, 0), req.offset + req.size)
        sz = logical_sizes[fid]
        _touch(fid)

        if fid in fast_tier:
            old_sz = fast_tier[fid]
            fast_used += sz - old_sz
            fast_tier[fid] = sz
            if fid not in tracker.tracked_files:
                tracker.add(fid, t, sz, raw=req.raw)
            tracker.update(fid, t, req.op_type, sz, req.offset)
            continue

        if fid not in tracker.tracked_files:
            tracker.add(fid, t, sz, raw=req.raw)
        tracker.update(fid, t, req.op_type, sz, req.offset)

        fast_tier[fid] = sz
        fast_used += sz

        while fast_used > fast_capacity and len(fast_tier) > 1:
            victim = _opt_pick_evict(t, exclude_fid=fid)
            if victim is None:
                break
            _do_eviction(victim, t)

    if not X_all:
        return np.empty((0, NUM_FEATURES)), np.empty(0), np.empty(0)

    X = np.concatenate(X_all)
    y = np.concatenate(y_all)
    g = np.concatenate(g_all)
    return X, y, g


def train_regressor(X, y, output_path):
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        num_leaves=63, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, verbose=-1,
    )
    model.fit(X, y)
    joblib.dump(model, output_path)
    print(f"Regressor saved to {output_path} ({X.shape[0]} samples)")
    return model


def train_ranker(X, y, groups, output_path):
    from lightgbm import LGBMRanker
    _, group_counts = np.unique(groups, return_counts=True)
    y_raw = np.expm1(y)
    relevance = np.zeros(len(y_raw), dtype=np.int32)
    relevance[y_raw > 1] = 1
    relevance[y_raw > 10] = 2
    relevance[y_raw > 60] = 3
    relevance[y_raw > 300] = 4
    model = LGBMRanker(
        objective="lambdarank", n_estimators=300, max_depth=8,
        learning_rate=0.05, num_leaves=63, min_child_samples=10,
        subsample=0.8, colsample_bytree=0.8, verbose=-1,
        label_gain=[0, 1, 3, 7, 15],
    )
    model.fit(X, relevance, group=group_counts)
    joblib.dump(model, output_path)
    print(f"Ranker saved to {output_path} ({X.shape[0]} samples, {len(group_counts)} groups)")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="traces/workload2/raw/cluster1_16TB_20240115_data-00000-of-00100 (1)")
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--warmup-rows", type=int, default=0)
    parser.add_argument("--fast-mb", type=int, default=512)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    print(f"Loading trace: {args.trace} (max {args.max_rows} rows)")
    requests = load_thesios_csv(args.trace, max_rows=args.max_rows, warmup_rows=args.warmup_rows)
    print(f"Loaded {len(requests)} requests")

    fast_cap = args.fast_mb * MB
    print(f"Generating imitation data (fast={args.fast_mb} MB)...")
    X, y, g = generate_imitation_data(requests, fast_capacity=fast_cap)
    print(f"Generated {X.shape[0]} samples in {int(g.max()) + 1 if len(g) else 0} groups")

    np.savez_compressed(f"{args.output_dir}/opt_imitation_data.npz", X=X, y=y, g=g)

    train_regressor(X, y, f"{args.output_dir}/opt_imitation_regressor.joblib")
    train_ranker(X, y, g, f"{args.output_dir}/opt_imitation_ranker.joblib")


if __name__ == "__main__":
    main()
