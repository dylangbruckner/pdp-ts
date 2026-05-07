"""Exponential-decay scoring policy (LRFU-like). No ML required."""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional

from .models import FastEntry, Operation, OpType, Request, Tier
from .policy import PolicyFunctions

def make_decay_policy(
    fast_capacity: float,
    alpha: float = 0.7,
    decay_rate: float = 0.05,
    recent_window: int = 32,
    fast_fill_threshold: float = 0.9,
    eviction_headroom: float = 0.05,
) -> PolicyFunctions:

    _access_times: Dict[str, deque] = {}
    _fast_sizes: Dict[str, int] = {}
    _fast_used: float = 0.0
    _last_time: float = 0.0

    def _score(fid: str, now: float) -> float:
        times = _access_times.get(fid)
        if not times:
            return 0.0
        dt = max(0, now - times[-1])
        recency = math.exp(max(-500, -decay_rate * dt))
        freq = sum(math.exp(max(-500, -decay_rate * (now - t))) for t in times if t <= now)
        return alpha * recency + (1 - alpha) * freq

    def _pick_evictions(needed: float, exclude: str = "") -> List[Operation]:
        nonlocal _fast_used
        target = needed + eviction_headroom * fast_capacity
        scored = sorted(
            ((fid, sz, _score(fid, _last_time)) for fid, sz in _fast_sizes.items() if fid != exclude),
            key=lambda x: x[2],
        )
        ops = []
        freed = 0.0
        for fid, sz, _ in scored:
            if freed >= target:
                break
            ops.append(Operation(OpType.EVICT, Tier.FAST, fid, sz, primary=False))
            freed += sz
        return ops

    def place_new_write(fid, size, offset, free, fe, fs):
        nonlocal _last_time
        _last_time = fs.get("_arrival_time", _last_time)
        _record_access(fid, _last_time)
        return [Operation(OpType.WRITE, Tier.FAST, fid, size, offset=offset, primary=True)]

    def handle_existing(request, in_fast, in_slow, free, fe, fs):
        nonlocal _last_time
        _last_time = request.arrival_time
        _record_access(request.file_id, _last_time)
        fid, sz = request.file_id, request.size

        if in_fast:
            return [Operation(request.op_type, Tier.FAST, fid, sz, offset=request.offset, primary=True)]

        ops = [Operation(request.op_type, Tier.SLOW, fid, sz, offset=request.offset, primary=True)]
        if free < sz:
            ops.extend(_pick_evictions(sz - free, exclude=fid))
        ops.append(Operation(OpType.WRITE, Tier.FAST, fid, sz, offset=request.offset, primary=False))
        return ops

    def evict_bytes(needed, fe, free, writing_fid):
        return _pick_evictions(needed, exclude=writing_fid)

    def bg_evict(fe, free, sim_time):
        # Don't overwrite _last_time — bg receives normalized SimPy time
        used = sum(e.size for e in fe.values())
        if used > fast_fill_threshold * fast_capacity:
            excess = used - fast_fill_threshold * fast_capacity
            return _pick_evictions(excess)
        return []

    def bg_promote(fe, free, sim_time):
        return []

    def _record_access(fid: str, t: float):
        if fid not in _access_times:
            _access_times[fid] = deque(maxlen=recent_window)
        _access_times[fid].append(t)

    def on_eviction(fid: str, tier: Tier):
        nonlocal _fast_used
        if tier == Tier.FAST and fid in _fast_sizes:
            _fast_used -= _fast_sizes.pop(fid)

    def on_write(fid: str, tier: Tier, size: int):
        nonlocal _fast_used
        if tier == Tier.FAST:
            old = _fast_sizes.get(fid, 0)
            _fast_sizes[fid] = size
            _fast_used += size - old

    return PolicyFunctions(
        place_new_write=place_new_write,
        handle_existing=handle_existing,
        evict_bytes=evict_bytes,
        bg_evict=bg_evict,
        bg_promote=bg_promote,
        bg_interval=5.0,
        on_eviction=on_eviction,
        on_write=on_write,
    )
