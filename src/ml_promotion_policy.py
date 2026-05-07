"""ML-guided promotion with LRU eviction.

Two variants:
  - hit_rate: predicts future_hit_bytes / file_size (cache efficiency)
  - response_time: predicts net response time savings per byte of cache used

Both train during warmup on observed reaccess patterns.
Promotion is gated: only promote when predicted value > threshold.
Eviction is always LRU.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Dict, List, Set

import numpy as np

from .models import FastEntry, Operation, OpType, Tier
from .policy import PolicyFunctions

FAST_READ_BW = 3.5e9
FAST_READ_LAT = 20e-6
SLOW_READ_BW = 200e6
SLOW_READ_LAT = 5e-3
FAST_WRITE_BW = 3.0e9


def make_ml_promotion_policy(
    fast_capacity: float,
    objective: str = "hit_rate",
    promote_threshold: float = 0.0,
    eviction_headroom: float = 0.05,
    fast_fill_threshold: float = 0.9,
    background_interval: float = 5.0,
    min_train_samples: int = 500,
    reaccess_window: float = 120.0,
) -> PolicyFunctions:

    _lru: OrderedDict = OrderedDict()
    _fast_sizes: Dict[str, int] = {}
    fast_used = [0.0]
    on_slow: Set[str] = set()
    sim_time = [0.0]

    _logical_sizes: Dict[str, int] = {}
    _io_sizes: Dict[str, List[int]] = defaultdict(list)
    _io_count: Dict[str, int] = defaultdict(int)
    _last_access: Dict[str, float] = {}
    _access_gaps: Dict[str, List[float]] = defaultdict(list)

    _model = [None]
    _trained = [False]

    _snapshots: Dict[str, tuple] = {}
    _train_X, _train_y = [], []

    def _update_stats(fid, size, offset, t):
        old_t = _last_access.get(fid)
        if old_t is not None:
            gap = t - old_t
            gaps = _access_gaps[fid]
            gaps.append(gap)
            if len(gaps) > 32:
                _access_gaps[fid] = gaps[-32:]
        _last_access[fid] = t
        _logical_sizes[fid] = max(_logical_sizes.get(fid, 0), offset + size)
        ios = _io_sizes[fid]
        ios.append(size)
        if len(ios) > 64:
            _io_sizes[fid] = ios[-64:]
        _io_count[fid] += 1

    def _features(fid, io_size):
        logical = _logical_sizes.get(fid, io_size)
        ios = _io_sizes.get(fid, [io_size])
        mean_io = np.mean(ios)
        count = _io_count.get(fid, 1)
        gaps = _access_gaps.get(fid, [])
        mean_gap = np.mean(gaps) if gaps else 999.0
        std_gap = np.std(gaps) if len(gaps) > 1 else 999.0
        min_gap = min(gaps) if gaps else 999.0

        t = sim_time[0]
        recency = t - _last_access.get(fid, t)
        promo_cost_ms = (logical / FAST_WRITE_BW) * 1000
        per_access_save_ms = ((mean_io / SLOW_READ_BW + SLOW_READ_LAT) -
                              (mean_io / FAST_READ_BW + FAST_READ_LAT)) * 1000
        size_ratio = logical / max(mean_io, 1)

        return np.array([
            np.log1p(logical),
            np.log1p(mean_io),
            size_ratio,
            count,
            np.log1p(recency),
            mean_gap,
            std_gap,
            min_gap,
            promo_cost_ms,
            per_access_save_ms,
            per_access_save_ms / max(promo_cost_ms, 0.001),
            len(gaps),
        ])

    def _record(fid, io_size, t):
        fv = _features(fid, io_size)
        logical = _logical_sizes.get(fid, io_size)
        mean_io = np.mean(_io_sizes.get(fid, [io_size]))
        _snapshots[fid] = (fv.copy(), t, logical, mean_io, 0, 0.0)

    def _label(fid, t, io_size):
        prev = _snapshots.pop(fid, None)
        if prev is None:
            return
        fv, snap_t, logical, mean_io, _, _ = prev
        gap = t - snap_t
        if gap > reaccess_window:
            if objective == "hit_rate":
                _train_X.append(fv)
                _train_y.append(0.0)
            else:
                _train_X.append(fv)
                _train_y.append(-((logical / FAST_WRITE_BW) * 1000))
            return

        est_future = min(reaccess_window / max(gap, 0.01), 200)

        if objective == "hit_rate":
            hit_bytes = est_future * mean_io
            label = hit_bytes / max(logical, 1)
        else:
            save_per_access = ((mean_io / SLOW_READ_BW + SLOW_READ_LAT) -
                               (mean_io / FAST_READ_BW + FAST_READ_LAT)) * 1000
            total_save = est_future * save_per_access
            promo_cost = (logical / FAST_WRITE_BW) * 1000
            label = (total_save - promo_cost) / max(logical / (1024**2), 0.001)

        _train_X.append(fv)
        _train_y.append(label)

    def _train():
        if len(_train_X) < min_train_samples:
            return
        from lightgbm import LGBMRegressor
        X = np.array(_train_X)
        y = np.array(_train_y)
        _model[0] = LGBMRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            min_child_samples=20, verbose=-1)
        _model[0].fit(X, y)
        _trained[0] = True
        pos = sum(1 for v in y if v > 0)
        print(f"[ml_promo_{objective}] Trained: {len(y)} samples, "
              f"positive={pos}/{len(y)} ({100*pos/len(y):.0f}%), "
              f"mean_label={np.mean(y):.3f}, median={np.median(y):.3f}")

    def _should_promote(fid, io_size):
        if not _trained[0]:
            return True
        fv = _features(fid, io_size).reshape(1, -1)
        pred = float(_model[0].predict(fv)[0])
        return pred > promote_threshold

    def _lru_evict(needed, fe, exclude=None):
        target = needed + eviction_headroom * fast_capacity
        ops, freed = [], 0.0
        for fid in list(_lru.keys()):
            if freed >= target:
                break
            if fid == exclude or fid not in fe:
                continue
            ops.append(Operation(OpType.EVICT, Tier.FAST, fid, fe[fid].size, primary=False))
            freed += fe[fid].size
        return ops

    def place_new_write(fid, size, offset, free, fe, fs):
        t = fs.get("_arrival_time", sim_time[0])
        sim_time[0] = t
        _update_stats(fid, size, offset, t)
        _record(fid, size, t)
        return [Operation(OpType.WRITE, Tier.FAST, fid, size, offset, primary=True)]

    def handle_existing(request, in_fast, in_slow, free, fe, fs):
        fid, sz = request.file_id, request.size
        sim_time[0] = request.arrival_time
        _update_stats(fid, sz, request.offset, request.arrival_time)
        _label(fid, request.arrival_time, sz)
        _record(fid, sz, request.arrival_time)

        if not _trained[0] and not request.is_warmup:
            _train()

        if in_fast:
            _lru[fid] = True
            _lru.move_to_end(fid)
            return [Operation(request.op_type, Tier.FAST, fid, sz,
                              offset=request.offset, primary=True)]

        ops = [Operation(request.op_type, Tier.SLOW, fid, sz,
                         offset=request.offset, primary=True)]

        if request.op_type == OpType.READ:
            if _should_promote(fid, sz):
                ops.append(Operation(OpType.WRITE, Tier.FAST, fid, sz,
                                     offset=request.offset, primary=False))
        elif request.op_type == OpType.WRITE and free >= sz:
            if _should_promote(fid, sz):
                return [Operation(OpType.WRITE, Tier.FAST, fid, sz,
                                  offset=request.offset, primary=True)]
        return ops

    def evict_bytes(needed, fe, free, writing_fid):
        return _lru_evict(needed, fe, exclude=writing_fid)

    def bg_evict(fe, free, st):
        if fast_used[0] <= fast_fill_threshold * fast_capacity:
            return []
        excess = fast_used[0] - fast_fill_threshold * fast_capacity
        return _lru_evict(excess, fe)

    def bg_promote(fe, free, st):
        return []

    def on_eviction(fid, tier):
        if tier == Tier.FAST:
            fast_used[0] -= _fast_sizes.pop(fid, 0)
            _lru.pop(fid, None)

    def on_write(fid, tier, size):
        if tier == Tier.SLOW:
            on_slow.add(fid)
        elif tier == Tier.FAST:
            old = _fast_sizes.get(fid, 0)
            fast_used[0] += size - old
            _fast_sizes[fid] = size
            _lru[fid] = True
            _lru.move_to_end(fid)

    return PolicyFunctions(
        place_new_write=place_new_write,
        handle_existing=handle_existing,
        evict_bytes=evict_bytes,
        bg_evict=bg_evict,
        bg_promote=bg_promote,
        bg_interval=background_interval,
        on_eviction=on_eviction,
        on_write=on_write,
    )
