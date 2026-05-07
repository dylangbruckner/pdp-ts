"""Online-retraining ML eviction policy."""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import joblib
import numpy as np

from .ml_features import FileTracker
from .models import FastEntry, Operation, OpType, Priority, Request, Tier
from .policy import PolicyFunctions


def make_online_ml_policy(
    fast_capacity: float,
    model_path: str,
    promote_on_miss: bool = True,
    eviction_headroom: float = 0.05,
    max_file_size_pct: float = 1.0,
    fast_fill_threshold: float = 1.0,
    background_interval: float = 5.0,
    recent_window: float = 60.0,
    retrain_interval: int = 10,
    min_retrain_samples: int = 500,
    buffer_capacity: int = 10000,
) -> PolicyFunctions:
    import lightgbm as lgb

    model_obj = [joblib.load(model_path)]
    tracker = FileTracker(recent_window=recent_window)
    fast_used = [0.0]
    _fast_sizes: Dict[str, int] = {}
    on_slow: Set[str] = set()
    sim_time = [0.0]

    _last_features: Dict[str, Tuple[float, np.ndarray]] = {}
    _train_X: deque = deque(maxlen=buffer_capacity)
    _train_y: deque = deque(maxlen=buffer_capacity)
    _bg_count = [0]
    _samples_since_retrain = [0]

    _model_params = dict(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        num_leaves=31, min_child_samples=50, verbose=-1,
    )

    def _too_large(size: int) -> bool:
        return size > max_file_size_pct * fast_capacity

    def _eviction_target(needed: float) -> float:
        return needed + eviction_headroom * fast_capacity

    def _record_label(fid: str, access_time: float):
        prev = _last_features.pop(fid, None)
        if prev is None:
            return
        ts, fv = prev
        label = np.log1p(access_time - ts)
        _train_X.append(fv)
        _train_y.append(label)
        _samples_since_retrain[0] += 1

    def _snapshot_features(fid: str, t: float):
        fv = tracker.feature_vector(fid, t)
        _last_features[fid] = (t, fv.copy())

    def _maybe_retrain():
        if _samples_since_retrain[0] < min_retrain_samples:
            return
        X = np.array(_train_X)
        y = np.array(_train_y)
        new_model = lgb.LGBMRegressor(**_model_params)
        new_model.fit(X, y)
        model_obj[0] = new_model
        _samples_since_retrain[0] = 0

    def _score_and_rank(
        fe: Dict[str, FastEntry], exclude: Optional[str] = None,
    ) -> List[Tuple[str, int, float]]:
        fe_fids = set(fe.keys())
        fids, X = tracker.all_feature_vectors(sim_time[0])
        if len(fids) == 0:
            return []
        mask = np.array([f in fe_fids for f in fids])
        fids_f = [f for f, m in zip(fids, mask) if m]
        X_f = X[mask]
        if len(fids_f) == 0:
            return []
        scores = model_obj[0].predict(X_f)
        ranked = sorted(zip(fids_f, [fe[f].size for f in fids_f], scores),
                        key=lambda x: -x[2])
        if exclude:
            ranked = [(f, s, sc) for f, s, sc in ranked if f != exclude]
        return ranked

    def place_new_write(fid, size, offset, free, fe, fs):
        if _too_large(size):
            return [Operation(OpType.WRITE, Tier.SLOW, fid, size, offset, primary=True)]
        raw = fs.get("_last_raw")
        tracker.add(fid, sim_time[0], size, raw=raw)
        tracker.update(fid, sim_time[0], OpType.WRITE, size)
        _snapshot_features(fid, sim_time[0])
        return [Operation(OpType.WRITE, Tier.FAST, fid, size, offset, primary=True)]

    def handle_existing(request, in_fast, in_slow, free, fe, fs):
        fid, sz = request.file_id, request.size
        sim_time[0] = request.arrival_time
        if fid not in tracker.tracked_files:
            tracker.add(fid, request.arrival_time, sz, raw=request.raw)
        tracker.update(fid, request.arrival_time, request.op_type, sz)

        _record_label(fid, request.arrival_time)
        _snapshot_features(fid, request.arrival_time)

        if _too_large(sz):
            return [Operation(request.op_type, Tier.SLOW, fid, sz,
                              offset=request.offset, primary=True)]
        if in_fast:
            return [Operation(request.op_type, Tier.FAST, fid, sz,
                              offset=request.offset, primary=True)]
        ops = [Operation(request.op_type, Tier.SLOW, fid, sz,
                         offset=request.offset, primary=True)]
        if request.op_type == OpType.READ and promote_on_miss:
            ops.append(Operation(OpType.WRITE, Tier.FAST, fid, sz,
                                 offset=request.offset, primary=False))
        elif request.op_type == OpType.WRITE and free >= sz:
            return [Operation(OpType.WRITE, Tier.FAST, fid, sz,
                              offset=request.offset, primary=True)]
        return ops

    def evict_bytes(needed, fe, free, writing_fid):
        ranked = _score_and_rank(fe, exclude=writing_fid)
        target = _eviction_target(needed)
        ops: List[Operation] = []
        freed = 0.0
        clean = [(f, s, sc) for f, s, sc in ranked if f in on_slow]
        dirty = [(f, s, sc) for f, s, sc in ranked if f not in on_slow]
        for fid, sz, _ in clean + dirty:
            if freed >= target:
                break
            ops.append(Operation(OpType.EVICT, Tier.FAST, fid, sz, primary=False))
            freed += sz
        return ops

    def bg_evict(fe, free, st):
        _bg_count[0] += 1
        if _bg_count[0] % retrain_interval == 0:
            _maybe_retrain()

        if fast_used[0] <= fast_fill_threshold * fast_capacity:
            return []
        excess = fast_used[0] - fast_fill_threshold * fast_capacity
        ranked = _score_and_rank(fe)
        target = _eviction_target(excess)
        ops: List[Operation] = []
        freed = 0.0
        for fid, sz, _ in ranked:
            if freed >= target:
                break
            ops.append(Operation(OpType.EVICT, Tier.FAST, fid, sz, primary=False))
            freed += sz
        return ops

    def bg_promote(fe, free, st):
        ranked = _score_and_rank(fe)
        return [
            Operation(OpType.WRITE, Tier.SLOW, fid, sz, primary=False)
            for fid, sz, _ in ranked if fid not in on_slow
        ]

    def on_eviction(fid, tier):
        if tier == Tier.FAST:
            fast_used[0] -= _fast_sizes.pop(fid, 0)
            tracker.remove(fid)
            _last_features.pop(fid, None)

    def on_write(fid, tier, size):
        if tier == Tier.SLOW:
            on_slow.add(fid)
        elif tier == Tier.FAST:
            old = _fast_sizes.get(fid, 0)
            fast_used[0] += size - old
            _fast_sizes[fid] = size
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
