"""Belady's OPT (optimal) eviction policy — oracle with full trace knowledge.

For files with no future access in the trace, falls back to LRU ordering
to avoid penalizing files that just happen to not reappear before trace end.
"""
from __future__ import annotations

import bisect
from collections import OrderedDict, defaultdict
from typing import Dict, List, Set

from .models import FastEntry, Operation, OpType, Request, Tier
from .policy import PolicyFunctions


def make_opt_policy(
    fast_capacity: float,
    requests: List[Request],
    promote_on_miss: bool = True,
    eviction_headroom: float = 0.05,
    fast_fill_threshold: float = 1.0,
    background_interval: float = 30.0,
    **kwargs,
) -> PolicyFunctions:

    sorted_reqs = sorted(requests, key=lambda r: r.arrival_time)
    _future: Dict[str, List[float]] = defaultdict(list)
    for r in sorted_reqs:
        _future[r.file_id].append(r.arrival_time)

    INF = float("inf")
    sim_time = [0.0]
    fast_used = [0.0]
    _fast_sizes: Dict[str, int] = {}
    on_slow: Set[str] = set()
    _lru: OrderedDict = OrderedDict()

    def _next_access_after(fid: str, t: float) -> float:
        times = _future.get(fid)
        if times is None:
            return INF
        idx = bisect.bisect_right(times, t)
        return times[idx] if idx < len(times) else INF

    def _touch(fid: str):
        _lru[fid] = True
        _lru.move_to_end(fid)

    def _pick_evictions(needed: float, fe: Dict[str, FastEntry],
                        exclude: str = None) -> List[Operation]:
        target = needed + eviction_headroom * fast_capacity
        t = sim_time[0]
        known_future = []
        no_future = []
        for fid, entry in fe.items():
            if fid == exclude:
                continue
            nxt = _next_access_after(fid, t)
            if nxt < INF:
                known_future.append((fid, entry.size, nxt))
            else:
                no_future.append(fid)

        # Evict known-future files coldest first (furthest next access)
        known_future.sort(key=lambda x: -x[2])
        # Evict no-future files in LRU order (coldest = front of _lru)
        lru_order = [f for f in _lru if f in set(no_future)]
        no_future_sorted = [(f, fe[f].size) for f in lru_order]

        ops: List[Operation] = []
        freed = 0.0
        # Prefer evicting no-future files first (LRU among them)
        for fid, sz in no_future_sorted:
            if freed >= target:
                break
            ops.append(Operation(OpType.EVICT, Tier.FAST, fid, sz, primary=False))
            freed += sz
        for fid, sz, _ in known_future:
            if freed >= target:
                break
            ops.append(Operation(OpType.EVICT, Tier.FAST, fid, sz, primary=False))
            freed += sz
        return ops

    def place_new_write(fid, size, offset, free, fe, fs):
        t = fs.get("_arrival_time")
        if t is not None:
            sim_time[0] = t
        _touch(fid)
        return [Operation(OpType.WRITE, Tier.FAST, fid, size, offset, primary=True)]

    def handle_existing(request, in_fast, in_slow, free, fe, fs):
        fid, sz = request.file_id, request.size
        sim_time[0] = request.arrival_time
        _touch(fid)

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
            _lru.pop(fid, None)

    def on_write(fid, tier, size):
        if tier == Tier.SLOW:
            on_slow.add(fid)
        elif tier == Tier.FAST:
            old = _fast_sizes.get(fid, 0)
            fast_used[0] += size - old
            _fast_sizes[fid] = size
            _touch(fid)

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
