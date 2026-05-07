"""ML-guided eviction policy using gradient boosted trees."""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import joblib
import numpy as np

from .ml_features import FileTracker
from .models import FastEntry, Operation, OpType, Priority, Request, Tier
from .policy import PolicyFunctions


def make_ml_policy(
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
    tracker = FileTracker(recent_window=recent_window)
    fast_used = [0.0]
    _fast_sizes: Dict[str, int] = {}  # ground-truth fast tier sizes (from evaluator callbacks)
    on_slow: Set[str] = set()
    sim_time = [0.0]

    def _too_large(size: int) -> bool:
        return size > max_file_size_pct * fast_capacity

    def _eviction_target(needed: float) -> float:
        return needed + eviction_headroom * fast_capacity

    def _score_and_rank(
        fe: Dict[str, FastEntry], exclude: Optional[str] = None,
    ) -> List[Tuple[str, int, float]]:
        """Return [(fid, size, coldness_score)] sorted coldest-first. Only scores files in fe."""
        fe_fids = set(fe.keys())
        fids, X = tracker.all_feature_vectors(sim_time[0])
        if len(fids) == 0:
            return []
        # filter to files actually in fast tier
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
        raw = fs.get("_last_raw")  # passed via evaluator snapshot if available
        tracker.add(fid, sim_time[0], size, raw=raw)
        tracker.update(fid, sim_time[0], OpType.WRITE, size)
        return [Operation(OpType.WRITE, Tier.FAST, fid, size, offset, primary=True)]

    def handle_existing(request, in_fast, in_slow, free, fe, fs):
        fid, sz = request.file_id, request.size
        sim_time[0] = request.arrival_time
        if fid not in tracker.tracked_files:
            tracker.add(fid, request.arrival_time, sz, raw=request.raw)
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
        # prefer clean files first (no drain cost)
        clean = [(f, s, sc) for f, s, sc in ranked if f in on_slow]
        dirty = [(f, s, sc) for f, s, sc in ranked if f not in on_slow]
        for fid, sz, _ in clean + dirty:
            if freed >= target:
                break
            ops.append(Operation(OpType.EVICT, Tier.FAST, fid, sz, primary=False))
            freed += sz
        return ops

    def bg_evict(fe, free, st):
        # don't overwrite sim_time — bg receives normalized SimPy time, tracker uses absolute
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
