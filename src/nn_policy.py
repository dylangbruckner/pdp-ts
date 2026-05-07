"""Neural network eviction policy trained online during warmup."""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .ml_features import FileTracker
from .models import FastEntry, Operation, OpType, Priority, Request, Tier
from .policy import PolicyFunctions


def make_nn_policy(
    fast_capacity: float,
    promote_on_miss: bool = True,
    eviction_headroom: float = 0.05,
    max_file_size_pct: float = 1.0,
    fast_fill_threshold: float = 0.9,
    background_interval: float = 5.0,
    recent_window: float = 60.0,
    hidden_layers: Tuple[int, ...] = (128, 64),
    max_iter: int = 500,
    use_scaler: bool = True,
    drop_categorical: bool = True,
) -> PolicyFunctions:
    tracker = FileTracker(recent_window=recent_window)
    fast_used = [0.0]
    _fast_sizes: Dict[str, int] = {}
    on_slow: Set[str] = set()
    sim_time = [0.0]

    _last_features: Dict[str, Tuple[float, np.ndarray]] = {}
    _train_X: List[np.ndarray] = []
    _train_y: List[float] = []
    _model = [None]
    _scaler = [None]
    _warmup_done = [False]

    _drop_idx = [9, 10] if drop_categorical else []

    def _mask_features(X: np.ndarray) -> np.ndarray:
        if not _drop_idx:
            return X
        if X.ndim == 1:
            X = X.copy()
            for i in _drop_idx:
                X[i] = 0.0
            return X
        X = X.copy()
        X[:, _drop_idx] = 0.0
        return X

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

    def _snapshot_features(fid: str, t: float):
        fv = tracker.feature_vector(fid, t)
        _last_features[fid] = (t, _mask_features(fv.copy()))

    def _train_model():
        if len(_train_X) < 50:
            return
        X = np.array(_train_X)
        y = np.array(_train_y)
        X = _mask_features(X)
        if use_scaler:
            sc = StandardScaler()
            X = sc.fit_transform(X)
            _scaler[0] = sc
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            max_iter=max_iter,
            learning_rate='adaptive',
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        )
        model.fit(X, y)
        _model[0] = model
        print(f"  NN trained: {len(_train_X)} samples, layers={hidden_layers}, "
              f"loss={model.loss_:.4f}")

    def _score_and_rank(
        fe: Dict[str, FastEntry], exclude: Optional[str] = None,
    ) -> List[Tuple[str, int, float]]:
        if _model[0] is None:
            return _lru_fallback(fe, exclude)
        fe_fids = set(fe.keys())
        fids, X = tracker.all_feature_vectors(sim_time[0])
        if len(fids) == 0:
            return []
        mask = np.array([f in fe_fids for f in fids])
        fids_f = [f for f, m in zip(fids, mask) if m]
        X_f = _mask_features(X[mask])
        if len(fids_f) == 0:
            return []
        if use_scaler and _scaler[0] is not None:
            X_f = _scaler[0].transform(X_f)
        scores = _model[0].predict(X_f)
        ranked = sorted(zip(fids_f, [fe[f].size for f in fids_f], scores),
                        key=lambda x: -x[2])
        if exclude:
            ranked = [(f, s, sc) for f, s, sc in ranked if f != exclude]
        return ranked

    def _lru_fallback(fe, exclude=None):
        items = []
        for fid, entry in fe.items():
            if fid == exclude:
                continue
            s = tracker._state.get(fid)
            t = s.last_access_time if s else 0.0
            items.append((fid, entry.size, -t))
        items.sort(key=lambda x: -x[2])
        return items

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
        tracker.update(fid, request.arrival_time, request.op_type, sz, request.offset)

        if not _warmup_done[0]:
            _record_label(fid, request.arrival_time)
            _snapshot_features(fid, request.arrival_time)
            if request.is_warmup is False and not _warmup_done[0]:
                _warmup_done[0] = True
                _train_model()

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
