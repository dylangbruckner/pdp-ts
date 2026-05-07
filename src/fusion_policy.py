"""Score fusion eviction: blend LRU recency rank with ML coldness rank."""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple

import joblib
import numpy as np

from .ml_features import FileTracker
from .models import FastEntry, Operation, OpType, Priority, Request, Tier
from .policy import PolicyFunctions


def make_fusion_policy(
    fast_capacity: float,
    model_path: str,
    alpha: float = 0.5,
    promote_on_miss: bool = True,
    eviction_headroom: float = 0.05,
    max_file_size_pct: float = 1.0,
    fast_fill_threshold: float = 1.0,
    background_interval: float = 5.0,
    recent_window: float = 60.0,
    blend_mode: str = "linear",
    append_lru_rank: bool = False,
) -> PolicyFunctions:
    model = joblib.load(model_path)
    tracker = FileTracker(recent_window=recent_window)
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

    def _fusion_rank(fe: Dict[str, FastEntry], exclude: Optional[str] = None) -> List[Tuple[str, int, float]]:
        fe_fids = set(fe.keys())
        fids, X = tracker.all_feature_vectors(sim_time[0])

        lru_list = list(_lru.keys())
        n = len(lru_list)
        if n == 0:
            return []

        lru_ranks = {fid: i / max(n - 1, 1) for i, fid in enumerate(lru_list)}

        if len(fids) > 0:
            mask = np.array([f in fe_fids for f in fids])
            fids_f = [f for f, m in zip(fids, mask) if m]
            X_f = X[mask]
            if len(fids_f) > 0 and X_f.shape[0] > 0:
                if append_lru_rank:
                    lru_col = np.array([lru_ranks.get(f, 1.0) for f in fids_f]).reshape(-1, 1)
                    X_f = np.hstack([X_f, lru_col])
                scores = model.predict(X_f)
                ml_scores = dict(zip(fids_f, scores))
            else:
                ml_scores = {}
        else:
            ml_scores = {}

        candidates = [fid for fid in lru_list if fid in fe_fids and fid != exclude]
        if not candidates:
            return []

        if ml_scores:
            ml_vals = [ml_scores[f] for f in candidates if f in ml_scores]
            if ml_vals:
                mn, mx = min(ml_vals), max(ml_vals)
                rng = mx - mn if mx > mn else 1.0
            else:
                mn, rng = 0.0, 1.0
        else:
            mn, rng = 0.0, 1.0

        result = []
        for fid in candidates:
            lr = lru_ranks.get(fid, 1.0)
            if fid in ml_scores:
                mr = (ml_scores[fid] - mn) / rng
            else:
                mr = lr

            if blend_mode == "geometric":
                fs = (lr ** alpha) * (mr ** (1 - alpha))
            else:
                fs = alpha * lr + (1 - alpha) * mr

            result.append((fid, _fast_sizes.get(fid, fe[fid].size), fs))

        result.sort(key=lambda x: -x[2])
        return result

    def _pick_evictions(needed: float, fe: Dict[str, FastEntry], exclude: Optional[str] = None) -> List[Operation]:
        ranked = _fusion_rank(fe, exclude)
        if not ranked:
            return _pick_evictions_lru(needed, exclude)
        target = _eviction_target(needed)
        ops: List[Operation] = []
        freed = 0.0
        for fid, sz, _ in ranked:
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
        raw = None
        if hasattr(fs, 'get'):
            raw = fs.get("_last_raw")
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
        return _pick_evictions(needed, fe, exclude=writing_fid)

    def bg_evict(fe, free, st):
        if fast_used[0] <= fast_fill_threshold * fast_capacity:
            return []
        excess = fast_used[0] - fast_fill_threshold * fast_capacity
        return _pick_evictions(excess, fe)

    def bg_promote(fe, free, st):
        ranked = _fusion_rank(fe)
        return [
            Operation(OpType.WRITE, Tier.SLOW, fid, sz, primary=False)
            for fid, sz, _ in ranked if fid not in on_slow
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
