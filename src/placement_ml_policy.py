"""ML-guided placement + promotion policy optimizing mean response time.

Placement classifier: for new writes, predict fast vs slow tier.
Promotion regressor: on slow-tier hit, estimate response time savings
  from promoting. Only promotes when savings exceed promotion cost.
  Key features: logical file size, mean I/O size, file_size/io_size ratio,
  estimated future accesses, promotion transfer time.
Eviction: standard LRU.
"""
from __future__ import annotations

import math
from collections import OrderedDict, deque
from typing import Dict, List, Set

import numpy as np

from .ml_features import FileTracker, NUM_FEATURES
from .models import FastEntry, Operation, OpType, Request, Tier
from .policy import PolicyFunctions


def make_placement_ml_policy(
    fast_capacity: float,
    promote_on_miss: bool = True,
    eviction_headroom: float = 0.05,
    fast_fill_threshold: float = 1.0,
    background_interval: float = 5.0,
    placement_threshold: float = 0.5,
    promotion_benefit_threshold: float = 0.0,
    min_train_samples: int = 500,
    fast_read_bw: float = 3.5e9,
    fast_read_lat: float = 20e-6,
    slow_read_bw: float = 200e6,
    slow_read_lat: float = 5e-3,
    fast_write_bw: float = 3.0e9,
) -> PolicyFunctions:

    tracker = FileTracker()
    _lru: OrderedDict = OrderedDict()
    _fast_sizes: Dict[str, int] = {}
    fast_used = [0.0]
    on_slow: Set[str] = set()
    sim_time = [0.0]

    # Per-file promotion-relevant stats
    _logical_sizes: Dict[str, int] = {}
    _io_sizes: Dict[str, List[int]] = {}
    _io_count: Dict[str, int] = {}

    _placement_model = [None]
    _promotion_model = [None]
    _trained = [False]

    # Training buffers
    _place_X: List[np.ndarray] = []
    _place_y: List[int] = []
    _promo_X: List[np.ndarray] = []
    _promo_y: List[float] = []  # regression target: time saved per future access (ms)

    _snapshot_features: Dict[str, tuple] = {}  # fid -> (features, time, logical_sz, mean_io)

    REACCESS_WINDOW = 60.0

    def _update_file_stats(fid: str, size: int, offset: int):
        _logical_sizes[fid] = max(_logical_sizes.get(fid, 0), offset + size)
        if fid not in _io_sizes:
            _io_sizes[fid] = []
        _io_sizes[fid].append(size)
        if len(_io_sizes[fid]) > 64:
            _io_sizes[fid] = _io_sizes[fid][-64:]
        _io_count[fid] = _io_count.get(fid, 0) + 1

    def _promo_features(fid: str, request_size: int) -> np.ndarray:
        """Build promotion-specific feature vector."""
        base = tracker.feature_vector(fid, sim_time[0]) if fid in tracker.tracked_files else np.zeros(NUM_FEATURES)
        logical_sz = _logical_sizes.get(fid, request_size)
        ios = _io_sizes.get(fid, [request_size])
        mean_io = np.mean(ios)
        std_io = np.std(ios) if len(ios) > 1 else 0.0
        count = _io_count.get(fid, 1)

        promo_cost_s = logical_sz / fast_write_bw
        per_access_saving_s = (logical_sz / slow_read_bw + slow_read_lat) - (logical_sz / fast_read_bw + fast_read_lat)
        io_to_file_ratio = mean_io / max(logical_sz, 1)

        extra = np.array([
            logical_sz,
            mean_io,
            std_io,
            io_to_file_ratio,
            promo_cost_s * 1000,         # promotion cost in ms
            per_access_saving_s * 1000,  # saving per access in ms
            count,
        ])
        return np.concatenate([base, extra])

    N_PROMO_FEATURES = NUM_FEATURES + 7

    def _record_snapshot(fid: str, request_size: int, t: float):
        fv = _promo_features(fid, request_size)
        logical_sz = _logical_sizes.get(fid, request_size)
        mean_io = np.mean(_io_sizes.get(fid, [request_size]))
        _snapshot_features[fid] = (fv.copy(), t, logical_sz, mean_io)

    def _label_snapshot(fid: str, t: float, request_size: int):
        prev = _snapshot_features.pop(fid, None)
        if prev is None:
            return
        fv, snap_t, logical_sz, mean_io = prev
        gap = t - snap_t

        n_future = REACCESS_WINDOW / max(gap, 0.001)
        n_future = min(n_future, 100)

        saving_per_access_ms = ((mean_io / slow_read_bw + slow_read_lat) -
                                (mean_io / fast_read_bw + fast_read_lat)) * 1000
        promo_cost_ms = (logical_sz / fast_write_bw) * 1000

        net_benefit_ms = n_future * saving_per_access_ms - promo_cost_ms

        _place_X.append(fv[:NUM_FEATURES])
        _place_y.append(1 if gap < REACCESS_WINDOW else 0)

        _promo_X.append(fv)
        _promo_y.append(net_benefit_ms)

    def _train_models():
        if len(_place_X) < min_train_samples:
            return
        from lightgbm import LGBMClassifier, LGBMRegressor

        X_place = np.array(_place_X)
        y_place = np.array(_place_y)
        _placement_model[0] = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            min_child_samples=20, verbose=-1,
        )
        _placement_model[0].fit(X_place, y_place)

        X_promo = np.array(_promo_X)
        y_promo = np.array(_promo_y)
        _promotion_model[0] = LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            min_child_samples=20, verbose=-1,
        )
        _promotion_model[0].fit(X_promo, y_promo)

        _trained[0] = True
        pos = sum(y_place)
        promo_pos = sum(1 for y in y_promo if y > 0)
        print(f"[placement_ml] Trained: {len(y_place)} samples, "
              f"place_hot={pos}/{len(y_place)}, "
              f"promo_beneficial={promo_pos}/{len(y_promo)}, "
              f"mean_benefit={np.mean(y_promo):.2f}ms")

    def _predict_place_hot(fid: str) -> float:
        if _placement_model[0] is None or fid not in tracker.tracked_files:
            return 0.5
        fv = tracker.feature_vector(fid, sim_time[0]).reshape(1, -1)
        return float(_placement_model[0].predict_proba(fv)[0, 1])

    def _predict_promo_benefit(fid: str, request_size: int) -> float:
        """Predict net time benefit (ms) of promoting this file."""
        if _promotion_model[0] is None:
            return 1.0
        fv = _promo_features(fid, request_size).reshape(1, -1)
        return float(_promotion_model[0].predict(fv)[0])

    def _lru_evict(needed: float, fe: Dict[str, FastEntry],
                   exclude: str = None) -> List[Operation]:
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
        if fid not in tracker.tracked_files:
            tracker.add(fid, t, size)
        tracker.update(fid, t, OpType.WRITE, size, offset)
        _update_file_stats(fid, size, offset)
        _record_snapshot(fid, size, t)

        if _trained[0]:
            if _predict_place_hot(fid) < placement_threshold:
                return [Operation(OpType.WRITE, Tier.SLOW, fid, size, offset, primary=True)]

        return [Operation(OpType.WRITE, Tier.FAST, fid, size, offset, primary=True)]

    def handle_existing(request, in_fast, in_slow, free, fe, fs):
        fid, sz = request.file_id, request.size
        sim_time[0] = request.arrival_time

        if fid not in tracker.tracked_files:
            tracker.add(fid, request.arrival_time, sz, raw=request.raw)
        tracker.update(fid, request.arrival_time, request.op_type, sz, request.offset)
        _update_file_stats(fid, sz, request.offset)

        _label_snapshot(fid, request.arrival_time, sz)
        _record_snapshot(fid, sz, request.arrival_time)

        if not _trained[0] and request.is_warmup is False:
            _train_models()

        if in_fast:
            _lru[fid] = True
            _lru.move_to_end(fid)
            return [Operation(request.op_type, Tier.FAST, fid, sz,
                              offset=request.offset, primary=True)]

        ops = [Operation(request.op_type, Tier.SLOW, fid, sz,
                         offset=request.offset, primary=True)]

        should_promote = promote_on_miss
        if _trained[0] and request.op_type == OpType.READ:
            benefit = _predict_promo_benefit(fid, sz)
            should_promote = benefit > promotion_benefit_threshold

        if request.op_type == OpType.READ and should_promote:
            ops.append(Operation(OpType.WRITE, Tier.FAST, fid, sz,
                                 offset=request.offset, primary=False))
        elif request.op_type == OpType.WRITE and free >= sz:
            if not _trained[0] or _predict_place_hot(fid) >= placement_threshold:
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
            if fid not in tracker.tracked_files:
                tracker.add(fid, sim_time[0], size)

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
