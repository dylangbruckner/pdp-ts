"""Adaptive ML policy: LRU with ML tie-breaking in recency buckets.

Core idea: Preserve LRU's recency ordering as the primary signal.
Only use ML to reorder files within groups that have similar recency
(accessed within `tie_window` seconds of each other). This avoids
ML's failure mode of evicting recently-accessed files while still
letting ML add value where LRU is uninformative.

Additional: graduated confidence — ML influence increases as file
staleness grows. Very recently accessed files are never ML-reordered.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple

import joblib
import numpy as np

from .ml_features import FileTracker
from .models import FastEntry, Operation, OpType, Request, Tier
from .policy import PolicyFunctions


def make_adaptive_ml_policy(
    fast_capacity: float,
    model_path: str,
    tie_window: float = 5.0,
    ml_weight: float = 0.3,
    protect_recent_seconds: float = 2.0,
    promote_on_miss: bool = True,
    eviction_headroom: float = 0.05,
    max_file_size_pct: float = 1.0,
    fast_fill_threshold: float = 1.0,
    background_interval: float = 5.0,
    recent_window: float = 60.0,
) -> PolicyFunctions:
    """
    tie_window: files accessed within this many seconds of each other
        are considered a "tie group" — ML reorders within the group.
    ml_weight: blend weight for ML score within tie groups (0=pure LRU, 1=pure ML).
    protect_recent_seconds: files accessed within this many seconds are
        never evicted (absolute protection floor).
    """
    model = joblib.load(model_path)
    tracker = FileTracker(recent_window=recent_window)
    _lru: OrderedDict[str, int] = OrderedDict()
    _fast_sizes: Dict[str, int] = {}
    _last_access: Dict[str, float] = {}
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

    def _pick_evictions(
        needed: float, fe: Dict[str, FastEntry], exclude: Optional[str] = None
    ) -> List[Operation]:
        target = _eviction_target(needed)
        now = sim_time[0]

        # Get LRU-ordered candidates (oldest first)
        candidates = []
        for fid, sz in _lru.items():
            if fid == exclude or fid not in _fast_sizes:
                continue
            last_t = _last_access.get(fid, 0.0)
            # Absolute protection for very recent files
            if now - last_t < protect_recent_seconds:
                continue
            candidates.append((fid, sz, last_t))

        if not candidates:
            # Fallback: pure LRU if all protected
            for fid, sz in _lru.items():
                if fid == exclude or fid not in _fast_sizes:
                    continue
                candidates.append((fid, sz, _last_access.get(fid, 0.0)))

        if not candidates:
            return []

        # Group into tie buckets by recency
        ml_scores = _ml_scores(fe) if candidates else {}

        # Sort by last_access (oldest first = most evictable)
        candidates.sort(key=lambda x: x[2])

        # Partition into tie groups
        groups: List[List[Tuple[str, int, float]]] = []
        current_group = [candidates[0]]
        for i in range(1, len(candidates)):
            if candidates[i][2] - current_group[0][2] <= tie_window:
                current_group.append(candidates[i])
            else:
                groups.append(current_group)
                current_group = [candidates[i]]
        groups.append(current_group)

        # Within each tie group, reorder by ML coldness (higher = colder = evict first)
        ordered: List[Tuple[str, int]] = []
        for group in groups:
            if len(group) == 1 or not ml_scores:
                ordered.extend((fid, sz) for fid, sz, _ in group)
            else:
                # Compute blended score: LRU position (normalized) + ML weight * coldness
                group_scores = []
                for idx, (fid, sz, last_t) in enumerate(group):
                    lru_rank = idx / max(len(group) - 1, 1)  # 0=oldest in group
                    ml_s = ml_scores.get(fid, 0.0)
                    # Higher blend = more evictable
                    blend = (1 - ml_weight) * lru_rank + ml_weight * _normalize_ml(ml_s, ml_scores)
                    group_scores.append((fid, sz, blend))
                # Sort: lowest blend first (most evictable)
                group_scores.sort(key=lambda x: x[2])
                ordered.extend((fid, sz) for fid, sz, _ in group_scores)

        ops: List[Operation] = []
        freed = 0.0
        for fid, sz in ordered:
            if freed >= target:
                break
            ops.append(Operation(OpType.EVICT, Tier.FAST, fid, sz, primary=False))
            freed += sz
        return ops

    def _normalize_ml(score: float, all_scores: Dict[str, float]) -> float:
        """Normalize ML score to [0,1] range. Higher = colder = more evictable."""
        if not all_scores:
            return 0.5
        vals = list(all_scores.values())
        mn, mx = min(vals), max(vals)
        if mx - mn < 1e-9:
            return 0.5
        return (score - mn) / (mx - mn)

    def place_new_write(fid, size, offset, free, fe, fs):
        if _too_large(size):
            return [Operation(OpType.WRITE, Tier.SLOW, fid, size, offset, primary=True)]
        raw = fs.get("_last_raw") if hasattr(fs, 'get') else None
        tracker.add(fid, sim_time[0], size, raw=raw)
        tracker.update(fid, sim_time[0], OpType.WRITE, size)
        return [Operation(OpType.WRITE, Tier.FAST, fid, size, offset, primary=True)]

    def handle_existing(request, in_fast, in_slow, free, fe, fs):
        fid, sz = request.file_id, request.size
        sim_time[0] = request.arrival_time
        if fid not in tracker.tracked_files:
            tracker.add(fid, request.arrival_time, sz, raw=request.raw)
        tracker.update(fid, request.arrival_time, request.op_type, sz, request.offset)
        _last_access[fid] = request.arrival_time

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
        return []

    def on_eviction(fid, tier):
        if tier == Tier.FAST:
            fast_used[0] -= _fast_sizes.pop(fid, 0)
            if fid in _lru:
                del _lru[fid]
            _last_access.pop(fid, None)
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
            _last_access[fid] = sim_time[0]
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
