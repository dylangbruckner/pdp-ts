"""Optimized OPT imitation data generator using heap-based eviction.
The original _opt_pick_evict does O(n) scan per eviction. This uses a
max-heap with lazy deletion for O(log n) amortized."""
from __future__ import annotations

import bisect
import heapq
import math
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple

import numpy as np

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ml_features import FileTracker, NUM_FEATURES
from src.models import OpType

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


def generate_imitation_data_fast(
    requests,
    fast_capacity: int = 512 * MB,
    bg_interval: float = 5.0,
    fill_threshold: float = 0.9,
    max_candidates_per_group: int = 500,
    sample_rate: int = 1,
    future_requests=None,
    progress_interval: int = 50000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    future = build_oracle(future_requests if future_requests is not None else requests)
    sorted_reqs = sorted(requests, key=lambda r: r.arrival_time)

    tracker = FileTracker()
    fast_tier: Dict[str, int] = {}
    fast_used = 0
    lru: OrderedDict = OrderedDict()

    # Heap: (-next_access_time, seq, fid) — max-heap via negation
    _heap: list = []
    _heap_seq = 0
    _na_cache: Dict[str, float] = {}  # fid -> cached next_access
    _na_seq: Dict[str, int] = {}  # fid -> seq when last pushed

    X_all, y_all, g_all = [], [], []
    group_id = 0
    last_bg_time = 0.0
    _eviction_counter = [0]

    def _touch(fid):
        lru[fid] = True
        lru.move_to_end(fid)

    def _push_na(fid, current_time):
        nonlocal _heap_seq
        na = next_access_after(future, fid, current_time)
        _na_cache[fid] = na
        _heap_seq += 1
        _na_seq[fid] = _heap_seq
        heapq.heappush(_heap, (-na, _heap_seq, fid))

    def _opt_pick_evict(current_time, exclude_fid=None):
        while _heap:
            neg_na, seq, fid = _heap[0]
            if fid not in fast_tier or fid == exclude_fid:
                heapq.heappop(_heap)
                continue
            if _na_seq.get(fid) != seq:
                heapq.heappop(_heap)
                continue
            na = -neg_na
            if na < current_time:
                heapq.heappop(_heap)
                _push_na(fid, current_time)
                continue
            return fid
        # Fallback: all entries stale/removed, shouldn't happen normally
        if fast_tier:
            return next(iter(fast_tier))
        return None

    def _collect_eviction_samples(current_time, evict_fid):
        nonlocal group_id
        _eviction_counter[0] += 1
        if _eviction_counter[0] % sample_rate != 0:
            return
        fids_in_tier = list(fast_tier.keys())
        if len(fids_in_tier) < 2:
            return
        if len(fids_in_tier) > max_candidates_per_group:
            lru_cold = [f for f in lru if f in fast_tier][:max_candidates_per_group // 2]
            remaining = [f for f in fids_in_tier if f not in set(lru_cold)]
            import random
            sample_n = max_candidates_per_group - len(lru_cold)
            sampled = random.sample(remaining, min(sample_n, len(remaining)))
            fids_in_tier = list(set(lru_cold + sampled))
            if evict_fid not in fids_in_tier:
                fids_in_tier.append(evict_fid)
        features, targets = [], []
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

    def _do_eviction(fid, current_time):
        nonlocal fast_used
        _collect_eviction_samples(current_time, fid)
        sz = fast_tier.pop(fid)
        fast_used -= sz
        tracker.remove(fid)
        lru.pop(fid, None)
        _na_cache.pop(fid, None)
        _na_seq.pop(fid, None)

    def _bg_evict(current_time):
        nonlocal fast_used
        if fast_used <= fill_threshold * fast_capacity:
            return
        while fast_used > fill_threshold * fast_capacity and fast_tier:
            victim = _opt_pick_evict(current_time)
            if victim is None:
                break
            _do_eviction(victim, current_time)

    logical_sizes: Dict[str, int] = {}
    total = len(sorted_reqs)
    for i, req in enumerate(sorted_reqs):
        if i > 0 and i % progress_interval == 0:
            pct = i / total * 100
            print(f"  OPT progress: {i}/{total} ({pct:.1f}%), "
                  f"fast_tier={len(fast_tier)} files, "
                  f"samples={sum(len(x) for x in X_all)}, "
                  f"groups={group_id}")

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
            _push_na(fid, t)
            continue

        if fid not in tracker.tracked_files:
            tracker.add(fid, t, sz, raw=req.raw)
        tracker.update(fid, t, req.op_type, sz, req.offset)

        fast_tier[fid] = sz
        fast_used += sz
        _push_na(fid, t)

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
    print(f"  OPT generation complete: {X.shape[0]} samples, {group_id} groups")
    return X, y, g
