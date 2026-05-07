"""Cost-aware promotion filter.

Wraps any PolicyFunctions to skip promotions where the file is too large
relative to the I/O size. Promotes only when estimated response time
savings exceed the promotion transfer cost.

Usage:
  lru = make_lru_policy(cap)
  policy = wrap_cost_aware(lru, fast_read_bw=3.5e9, slow_read_bw=200e6, ...)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .models import Operation, OpType, Tier
from .policy import PolicyFunctions


def wrap_cost_aware(
    base: PolicyFunctions,
    max_file_io_ratio: float = 50.0,
    min_accesses_to_promote: int = 2,
    fast_read_bw: float = 3.5e9,
    fast_read_lat: float = 20e-6,
    slow_read_bw: float = 200e6,
    slow_read_lat: float = 5e-3,
    fast_write_bw: float = 3.0e9,
) -> PolicyFunctions:
    """Wrap a policy with cost-aware promotion filtering.

    Blocks background WRITE FAST (promotions) when:
      1. logical_file_size / mean_io_size > max_file_io_ratio, OR
      2. file has fewer than min_accesses_to_promote accesses
    """

    _logical_sizes: Dict[str, int] = {}
    _io_sizes: Dict[str, List[int]] = defaultdict(list)
    _access_count: Dict[str, int] = defaultdict(int)

    def _update_stats(fid: str, size: int, offset: int):
        _logical_sizes[fid] = max(_logical_sizes.get(fid, 0), offset + size)
        ios = _io_sizes[fid]
        ios.append(size)
        if len(ios) > 64:
            _io_sizes[fid] = ios[-64:]
        _access_count[fid] += 1

    def _should_promote(fid: str, io_size: int) -> bool:
        logical = _logical_sizes.get(fid, io_size)
        if _access_count[fid] < min_accesses_to_promote:
            return False

        ios = _io_sizes.get(fid, [io_size])
        mean_io = sum(ios) / len(ios)
        ratio = logical / max(mean_io, 1)
        if ratio > max_file_io_ratio:
            return False

        return True

    def _filter_promotions(ops: List[Operation]) -> List[Operation]:
        result = []
        for op in ops:
            if (op.op_type == OpType.WRITE and op.tier == Tier.FAST
                    and not op.primary):
                if not _should_promote(op.file_id, op.size):
                    continue
            result.append(op)
        return result

    orig_place = base.place_new_write
    orig_handle = base.handle_existing

    def place_new_write(fid, size, offset, free, fe, fs):
        _update_stats(fid, size, offset)
        ops = orig_place(fid, size, offset, free, fe, fs)
        return _filter_promotions(ops)

    def handle_existing(request, in_fast, in_slow, free, fe, fs):
        _update_stats(request.file_id, request.size, request.offset)
        ops = orig_handle(request, in_fast, in_slow, free, fe, fs)
        return _filter_promotions(ops)

    return PolicyFunctions(
        place_new_write=place_new_write,
        handle_existing=handle_existing,
        evict_bytes=base.evict_bytes,
        bg_evict=base.bg_evict,
        bg_promote=base.bg_promote,
        bg_interval=base.bg_interval,
        on_eviction=base.on_eviction,
        on_write=base.on_write,
    )
