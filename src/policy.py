"""
PolicyFunctions: the 5-callable bundle that drives all placement decisions.

Evaluator calls:
  place_new_write  — WRITE for a file not yet in fast or slow
  handle_existing  — any request for a file already known to the system
  evict_bytes      — make space on demand before a primary WRITE FAST
  bg_evict         — background eviction (periodic)
  bg_promote       — background promotion / write-behind (periodic)

Callbacks (optional):
  on_eviction(file_id, tier)           called after evaluator removes a file from a tier
  on_write(file_id, tier, size)        called after every successful write completes

Factory functions (make_lru_policy, etc.) wrap the existing DPA classes
and wire the callbacks so state stays in sync.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .models import FastEntry, Operation, OpType, Priority, Request, Tier


# type aliases for readability
FsSnapshot = Dict[str, Dict[str, Optional[int]]]
FastEntries = Dict[str, FastEntry]

PlaceNewWriteFn = Callable[[str, int, int, float, FastEntries, FsSnapshot], List[Operation]]
HandleExistingFn = Callable[[Request, bool, bool, float, FastEntries, FsSnapshot], List[Operation]]
EvictBytesFn = Callable[[float, FastEntries, float, str], List[Operation]]
BgFn = Callable[[FastEntries, float, float], List[Operation]]


def _noop_bg(fe, free, sim_time):
    return []


@dataclass
class PolicyFunctions:
    place_new_write: PlaceNewWriteFn
    handle_existing: HandleExistingFn
    evict_bytes: EvictBytesFn
    bg_evict: BgFn
    bg_promote: BgFn
    bg_interval: float = 5.0
    # optional state-sync callbacks; set by factory functions
    on_eviction: Optional[Callable[[str, Tier], None]] = field(default=None, repr=False)
    on_write: Optional[Callable[[str, Tier, int], None]] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_lru_policy(fast_capacity: float, **kwargs) -> PolicyFunctions:
    """Wrap LRUPolicy in the PolicyFunctions interface."""
    from .data_placement import LRUPolicy
    p = LRUPolicy(fast_capacity, **kwargs)

    def place_new_write(fid, size, offset, free, fe, fs):
        req = Request(0.0, OpType.WRITE, fid, offset, size)
        return p.on_request(req, free)

    def handle_existing(request, in_fast, in_slow, free, fe, fs):
        return p.on_request(request, free)

    def evict_bytes(needed, fe, free, writing_fid):
        return p._pick_evictions(needed)

    def bg_evict(fe, free, sim_time):
        if p._fast_used > p.fast_fill_threshold * p.fast_capacity:
            excess = p._fast_used - p.fast_fill_threshold * p.fast_capacity
            return p._pick_evictions(excess)
        return []

    def bg_promote(fe, free, sim_time):
        return p._pick_write_behind() if p.write_behind else []

    return PolicyFunctions(
        place_new_write=place_new_write,
        handle_existing=handle_existing,
        evict_bytes=evict_bytes,
        bg_evict=bg_evict,
        bg_promote=bg_promote,
        bg_interval=p._background_interval,
        on_eviction=lambda fid, tier: p.notify_eviction(fid, tier),
        on_write=lambda fid, tier, size: p.notify_write(fid, tier, size),
    )


def make_size_aware_lru_policy(fast_capacity: float, **kwargs) -> PolicyFunctions:
    from .data_placement import SizeAwareLRUPolicy
    p = SizeAwareLRUPolicy(fast_capacity, **kwargs)

    def place_new_write(fid, size, offset, free, fe, fs):
        req = Request(0.0, OpType.WRITE, fid, offset, size)
        return p.on_request(req, free)

    def handle_existing(request, in_fast, in_slow, free, fe, fs):
        return p.on_request(request, free)

    def evict_bytes(needed, fe, free, writing_fid):
        return p._pick_evictions(needed)

    def bg_evict(fe, free, sim_time):
        if p._fast_used > p.fast_fill_threshold * p.fast_capacity:
            excess = p._fast_used - p.fast_fill_threshold * p.fast_capacity
            return p._pick_evictions(excess)
        return []

    def bg_promote(fe, free, sim_time):
        return p._pick_write_behind() if p.write_behind else []

    return PolicyFunctions(
        place_new_write=place_new_write,
        handle_existing=handle_existing,
        evict_bytes=evict_bytes,
        bg_evict=bg_evict,
        bg_promote=bg_promote,
        bg_interval=p._background_interval,
        on_eviction=lambda fid, tier: p.notify_eviction(fid, tier),
        on_write=lambda fid, tier, size: p.notify_write(fid, tier, size),
    )


def make_lfu_policy(fast_capacity: float, **kwargs) -> PolicyFunctions:
    from .data_placement import LFUPolicy
    p = LFUPolicy(fast_capacity, **kwargs)

    def place_new_write(fid, size, offset, free, fe, fs):
        req = Request(0.0, OpType.WRITE, fid, offset, size)
        return p.on_request(req, free)

    def handle_existing(request, in_fast, in_slow, free, fe, fs):
        return p.on_request(request, free)

    def evict_bytes(needed, fe, free, writing_fid):
        return p._pick_evictions(needed)

    def bg_evict(fe, free, sim_time):
        if p._fast_used > p.fast_fill_threshold * p.fast_capacity:
            excess = p._fast_used - p.fast_fill_threshold * p.fast_capacity
            return p._pick_evictions(excess)
        return []

    def bg_promote(fe, free, sim_time):
        return p._pick_write_behind() if p.write_behind else []

    return PolicyFunctions(
        place_new_write=place_new_write,
        handle_existing=handle_existing,
        evict_bytes=evict_bytes,
        bg_evict=bg_evict,
        bg_promote=bg_promote,
        bg_interval=p._background_interval,
        on_eviction=lambda fid, tier: p.notify_eviction(fid, tier),
        on_write=lambda fid, tier, size: p.notify_write(fid, tier, size),
    )


def make_s3fifo_policy(fast_capacity: float, **kwargs) -> PolicyFunctions:
    from .data_placement import S3FIFOPolicy
    p = S3FIFOPolicy(fast_capacity, **kwargs)

    def place_new_write(fid, size, offset, free, fe, fs):
        req = Request(0.0, OpType.WRITE, fid, offset, size)
        return p.on_request(req, free)

    def handle_existing(request, in_fast, in_slow, free, fe, fs):
        return p.on_request(request, free)

    def evict_bytes(needed, fe, free, writing_fid):
        # S3FIFO evicts from small first, then main
        ops = p._drain_small(needed)
        freed = sum(fe[o.file_id].size for o in ops
                    if o.op_type == OpType.EVICT and o.file_id in fe)
        if freed < needed:
            ops.extend(p._drain_main(needed - freed))
        return ops

    def bg_evict(fe, free, sim_time):
        ops = []
        min_free = p._small_cap * p.SMALL_MIN_FREE
        if p._small_used > p._small_cap - min_free:
            ops.extend(p._drain_small(p._small_used - (p._small_cap - min_free)))
        total = p._small_used + p._main_used
        if total > p.fast_fill_threshold * p.fast_capacity:
            ops.extend(p._drain_main(total - p.fast_fill_threshold * p.fast_capacity))
        p._fast_used = p._small_used + p._main_used
        return ops

    def bg_promote(fe, free, sim_time):
        ops = p._pick_write_behind() if p.write_behind else []
        return ops

    return PolicyFunctions(
        place_new_write=place_new_write,
        handle_existing=handle_existing,
        evict_bytes=evict_bytes,
        bg_evict=bg_evict,
        bg_promote=bg_promote,
        bg_interval=p._background_interval,
        on_eviction=lambda fid, tier: p.notify_eviction(fid, tier),
        on_write=lambda fid, tier, size: p.notify_write(fid, tier, size),
    )


def make_ml_policy(fast_capacity: float, model_path: str, **kwargs) -> PolicyFunctions:
    """ML-guided eviction using a trained coldness model."""
    from .ml_policy import make_ml_policy as _make
    return _make(fast_capacity, model_path, **kwargs)


def make_hybrid_policy(fast_capacity: float, model_path: str, **kwargs) -> PolicyFunctions:
    """LRU eviction with ML-based protection of predicted-hot files."""
    from .hybrid_policy import make_hybrid_policy as _make
    return _make(fast_capacity, model_path, **kwargs)


def make_online_ml_policy(fast_capacity: float, model_path: str, **kwargs) -> PolicyFunctions:
    """ML eviction with periodic online retraining from observed access patterns."""
    from .online_ml_policy import make_online_ml_policy as _make
    return _make(fast_capacity, model_path, **kwargs)


def make_opt_policy(fast_capacity: float, requests, **kwargs) -> PolicyFunctions:
    """Belady's OPT — oracle eviction using full trace pre-scan."""
    from .opt_policy import make_opt_policy as _make
    return _make(fast_capacity, requests, **kwargs)


def make_fusion_policy(fast_capacity: float, model_path: str, **kwargs) -> PolicyFunctions:
    """Score fusion: blend LRU recency rank with ML coldness rank."""
    from .fusion_policy import make_fusion_policy as _make
    return _make(fast_capacity, model_path, **kwargs)


def make_placement_ml_policy(fast_capacity: float, **kwargs) -> PolicyFunctions:
    """ML classifiers for placement (fast vs slow) and promotion (promote on miss or not)."""
    from .placement_ml_policy import make_placement_ml_policy as _make
    return _make(fast_capacity, **kwargs)


def make_adaptive_ml_policy(fast_capacity: float, model_path: str, **kwargs) -> PolicyFunctions:
    """LRU with ML tie-breaking in recency buckets."""
    from .adaptive_ml_policy import make_adaptive_ml_policy as _make
    return _make(fast_capacity, model_path, **kwargs)
