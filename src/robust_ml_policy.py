"""ML eviction policy using only robust (generalizable) features."""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple

import joblib
import numpy as np

from .robust_features import RobustTracker
from .models import FastEntry, Operation, OpType, Tier
from .policy import PolicyFunctions


def make_robust_ml_policy(
    fast_capacity: float,
    model_path: str,
    promote_on_miss: bool = True,
    eviction_headroom: float = 0.05,
    max_file_size_pct: float = 1.0,
    fast_fill_threshold: float = 1.0,
    background_interval: float = 5.0,
    recent_window: float = 60.0,
) -> PolicyFunctions:
    model = joblib.load(model_path)
    tracker = RobustTracker(recent_window=recent_window)
    fast_used = [0.0]
    _fast_sizes: Dict[str, int] = {}
    on_slow: Set[str] = set()
    sim_time = [0.0]

    def _too_large(size: int) -> bool:
        return size > max_file_size_pct * fast_capacity

    def _eviction_target(needed: float) -> float:
        return needed + eviction_headroom * fast_capacity

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
        scores = model.predict(X_f)
        ranked = sorted(zip(fids_f, [fe[f].size for f in fids_f], scores),
                        key=lambda x: -x[2])
        if exclude:
            ranked = [(f, s, sc) for f, s, sc in ranked if f != exclude]
        return ranked

    def place_new_write(fid, size, offset, free, fe, fs):
        if _too_large(size):
            return [Operation(OpType.WRITE, Tier.SLOW, fid, size, offset, primary=True)]
        tracker.add(fid, sim_time[0], size)
        tracker.update(fid, sim_time[0], OpType.WRITE, size)
        return [Operation(OpType.WRITE, Tier.FAST, fid, size, offset, primary=True)]

    def handle_existing(request, in_fast, in_slow, free, fe, fs):
        fid, sz = request.file_id, request.size
        sim_time[0] = request.arrival_time
        if fid not in tracker.tracked_files:
            tracker.add(fid, request.arrival_time, sz)
        tracker.update(fid, request.arrival_time, request.op_type, sz, request.offset)

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


def make_robust_hybrid_policy(
    fast_capacity: float,
    model_path: str,
    protect_percentile: float = 0.25,
    promote_on_miss: bool = True,
    eviction_headroom: float = 0.05,
    max_file_size_pct: float = 1.0,
    fast_fill_threshold: float = 1.0,
    background_interval: float = 5.0,
    recent_window: float = 60.0,
) -> PolicyFunctions:
    model = joblib.load(model_path)
    tracker = RobustTracker(recent_window=recent_window)
    _lru: OrderedDict[str, int] = OrderedDict()
    _fast_sizes: Dict[str, int] = {}
    fast_used = [0.0]
    on_slow: Set[str] = set()
    sim_time = [0.0]

    def _too_large(size: int) -> bool:
        return size > max_file_size_pct * fast_capacity

    def _eviction_target(needed: float) -> float:
        return needed + eviction_headroom * fast_capacity

    def _touch(fid: str, size: int) -> None:
        if fid in _lru:
            _lru[fid] = size
            _lru.move_to_end(fid)

    def _ml_scores(fe: Dict[str, FastEntry]) -> Dict[str, float]:
        fe_fids = set(fe.keys())
        fids, X = tracker.all_feature_vectors(sim_time[0])
        if len(fids) == 0:
            return {}
        mask = np.array([f in fe_fids for f in fids])
        fids_f = [f for f, m in zip(fids, mask) if m]
        X_f = X[mask]
        if len(fids_f) == 0:
            return {}
        scores = model.predict(X_f)
        return dict(zip(fids_f, scores))

    def _pick_evictions_hybrid(
        needed: float, fe: Dict[str, FastEntry], exclude: Optional[str] = None,
    ) -> List[Operation]:
        scores = _ml_scores(fe)
        if not scores:
            return _pick_evictions_lru(needed, exclude)
        all_scores = list(scores.values())
        threshold = np.percentile(all_scores, protect_percentile * 100)
        is_protected = lambda s: s < threshold

        target = _eviction_target(needed)
        ops: List[Operation] = []
        freed = 0.0
        fallback: List[Tuple[str, int]] = []

        for fid, sz in list(_lru.items()):
            if freed >= target:
                break
            if fid == exclude or fid not in _fast_sizes:
                continue
            score = scores.get(fid)
            if score is not None and is_protected(score):
                fallback.append((fid, sz))
                continue
            ops.append(Operation(OpType.EVICT, Tier.FAST, fid, sz, primary=False))
            freed += sz

        for fid, sz in fallback:
            if freed >= target:
                break
            ops.append(Operation(OpType.EVICT, Tier.FAST, fid, sz, primary=False))
            freed += sz
        return ops

    def _pick_evictions_lru(needed: float, exclude: Optional[str] = None) -> List[Operation]:
        target = _eviction_target(needed)
        ops: List[Operation] = []
        freed = 0.0
        for fid, sz in list(_lru.items()):
            if freed >= target:
                break
            if fid == exclude:
                continue
            ops.append(Operation(OpType.EVICT, Tier.FAST, fid, sz, primary=False))
            freed += sz
        return ops

    def place_new_write(fid, size, offset, free, fe, fs):
        if _too_large(size):
            return [Operation(OpType.WRITE, Tier.SLOW, fid, size, offset, primary=True)]
        tracker.add(fid, sim_time[0], size)
        tracker.update(fid, sim_time[0], OpType.WRITE, size)
        return [Operation(OpType.WRITE, Tier.FAST, fid, size, offset, primary=True)]

    def handle_existing(request, in_fast, in_slow, free, fe, fs):
        fid, sz = request.file_id, request.size
        sim_time[0] = request.arrival_time
        if fid not in tracker.tracked_files:
            tracker.add(fid, request.arrival_time, sz)
        tracker.update(fid, request.arrival_time, request.op_type, sz, request.offset)

        if _too_large(sz):
            return [Operation(request.op_type, Tier.SLOW, fid, sz,
                              offset=request.offset, primary=True)]
        if in_fast:
            _touch(fid, sz)
            return [Operation(request.op_type, Tier.FAST, fid, sz,
                              offset=request.offset, primary=True)]

        ops = [Operation(request.op_type, Tier.SLOW, fid, sz,
                         offset=request.offset, primary=True)]
        if request.op_type == OpType.READ and promote_on_miss:
            _touch(fid, sz)
            ops.append(Operation(OpType.WRITE, Tier.FAST, fid, sz,
                                 offset=request.offset, primary=False))
        elif request.op_type == OpType.WRITE and free >= sz:
            _touch(fid, sz)
            return [Operation(OpType.WRITE, Tier.FAST, fid, sz,
                              offset=request.offset, primary=True)]
        return ops

    def evict_bytes(needed, fe, free, writing_fid):
        return _pick_evictions_hybrid(needed, fe, exclude=writing_fid)

    def bg_evict(fe, free, st):
        if fast_used[0] <= fast_fill_threshold * fast_capacity:
            return []
        excess = fast_used[0] - fast_fill_threshold * fast_capacity
        return _pick_evictions_hybrid(excess, fe)

    def bg_promote(fe, free, st):
        scores = _ml_scores(fe)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [
            Operation(OpType.WRITE, Tier.SLOW, fid, _lru.get(fid, 0), primary=False)
            for fid, _ in ranked
            if fid not in on_slow and fid in _lru
        ]

    def on_eviction(fid, tier):
        if tier == Tier.FAST:
            fast_used[0] -= _fast_sizes.pop(fid, 0)
            if fid in _lru:
                del _lru[fid]
            tracker.remove(fid)

    def on_write(fid, tier, size):
        if tier == Tier.SLOW:
            on_slow.add(fid)
        elif tier == Tier.FAST:
            old = _fast_sizes.get(fid, 0)
            fast_used[0] += size - old
            _fast_sizes[fid] = size
            _lru[fid] = size
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
