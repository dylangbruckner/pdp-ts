"""ML eviction policy that trains during warmup phase of eval shard."""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .ml_features import FileTracker, NUM_FEATURES
from .models import FastEntry, Operation, OpType, Priority, Request, Tier
from .policy import PolicyFunctions


def make_warmup_ml_policy(
    fast_capacity: float,
    promote_on_miss: bool = True,
    eviction_headroom: float = 0.05,
    max_file_size_pct: float = 1.0,
    fast_fill_threshold: float = 0.9,
    background_interval: float = 5.0,
    recent_window: float = 60.0,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    min_child_samples: int = 50,
    use_ranker: bool = False,
    n_relevance_buckets: int = 5,
) -> PolicyFunctions:
    import lightgbm as lgb

    tracker = FileTracker(recent_window=recent_window)
    fast_used = [0.0]
    _fast_sizes: Dict[str, int] = {}
    on_slow: Set[str] = set()
    sim_time = [0.0]

    _last_features: Dict[str, Tuple[float, np.ndarray]] = {}
    _train_X: List[np.ndarray] = []
    _train_y: List[float] = []
    _trained = [False]
    _warmup_ended = [False]
    _model = [None]

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
        _last_features[fid] = (t, fv.copy())

    def _train_model():
        if len(_train_X) < 100:
            print(f"[warmup_ml] Not enough samples ({len(_train_X)}), skipping training")
            return
        X = np.array(_train_X)
        y = np.array(_train_y)
        print(f"[warmup_ml] Training on {len(X)} samples")

        if use_ranker:
            buckets = np.quantile(y, np.linspace(0, 1, n_relevance_buckets + 1)[1:-1])
            y_rel = np.digitize(y, buckets).astype(int)
            model = lgb.LGBMRanker(
                objective="lambdarank",
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_child_samples=min_child_samples,
                verbose=-1,
            )
            model.fit(X, y_rel, group=[len(X)])
        else:
            model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_child_samples=min_child_samples,
                verbose=-1,
            )
            model.fit(X, y)

        _model[0] = model
        _trained[0] = True
        print(f"[warmup_ml] Model trained ({'ranker' if use_ranker else 'regressor'})")

    def _score_and_rank(
        fe: Dict[str, FastEntry], exclude: Optional[str] = None,
    ) -> List[Tuple[str, int, float]]:
        if _model[0] is None:
            return []
        fe_fids = set(fe.keys())
        fids, X = tracker.all_feature_vectors(sim_time[0])
        if len(fids) == 0:
            return []
        mask = np.array([f in fe_fids for f in fids])
        fids_f = [f for f, m in zip(fids, mask) if m]
        X_f = X[mask]
        if len(fids_f) == 0:
            return []
        if X_f.shape[0] > 5000:
            chunks = [X_f[i:i+5000] for i in range(0, X_f.shape[0], 5000)]
            scores = np.concatenate([_model[0].predict(c) for c in chunks])
        else:
            scores = _model[0].predict(X_f)
        ranked = sorted(zip(fids_f, [fe[f].size for f in fids_f], scores),
                        key=lambda x: -x[2])
        if exclude:
            ranked = [(f, s, sc) for f, s, sc in ranked if f != exclude]
        return ranked

    def _lru_evict_order(fe: Dict[str, FastEntry], exclude: Optional[str] = None):
        """Fallback LRU ordering before model is trained."""
        fids_in_fe = set(fe.keys())
        items = []
        for fid, state in tracker.tracked_files.items():
            if fid in fids_in_fe and fid != exclude:
                items.append((fid, fe[fid].size, -state.last_access_time))
        items.sort(key=lambda x: x[2], reverse=True)
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

        if not _warmup_ended[0] and not request.is_warmup:
            _warmup_ended[0] = True
            _train_model()
            _train_X.clear()
            _train_y.clear()
            _last_features.clear()

        if fid not in tracker.tracked_files:
            tracker.add(fid, request.arrival_time, sz, raw=request.raw)
        tracker.update(fid, request.arrival_time, request.op_type, sz, request.offset)

        if _warmup_ended[0] and not _trained[0]:
            pass
        elif not _warmup_ended[0]:
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
        if _trained[0]:
            ranked = _score_and_rank(fe, exclude=writing_fid)
        else:
            ranked = _lru_evict_order(fe, exclude=writing_fid)
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
        if _trained[0]:
            ranked = _score_and_rank(fe)
        else:
            ranked = _lru_evict_order(fe)
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
        if _trained[0]:
            ranked = _score_and_rank(fe)
        else:
            ranked = _lru_evict_order(fe)
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
