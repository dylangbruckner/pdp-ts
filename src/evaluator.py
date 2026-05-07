"""
Trace-driven two-tier storage evaluator using SimPy event-driven simulation.

Fast tier: limited capacity, faster bandwidth/latency.
Slow tier: unlimited capacity, slower bandwidth/latency.

Data placement decisions are delegated to a PolicyFunctions bundle of 5 pluggable callables.
"""
from __future__ import annotations

import time as wall_clock
from typing import Dict, List, Optional
import simpy

from .config import EvaluatorConfig, TierConfig
from .policy import PolicyFunctions
from .metrics import MetricsCollector
from .models import (
    FastEntry, Operation, OpType, Priority, Request, RequestRecord, Tier,
)

# type alias for the filesystem snapshot passed to policy functions
FsSnapshot = Dict[str, Dict[str, Optional[int]]]


class StorageTier:
    def __init__(self, env: simpy.Environment, cfg: TierConfig):
        self.cfg = cfg
        self.read_res = simpy.PriorityResource(env, capacity=cfg.max_concurrent_reads)
        self.write_res = simpy.PriorityResource(env, capacity=cfg.max_concurrent_writes)

    def read_duration(self, size: int) -> float:
        return self.cfg.read_latency + size / self.cfg.read_bandwidth

    def write_duration(self, size: int) -> float:
        return self.cfg.write_latency + size / self.cfg.write_bandwidth


class Evaluator:
    def __init__(
        self,
        config: EvaluatorConfig,
        policy: PolicyFunctions,
        requests: List[Request],
    ):
        self.config = config
        self.policy = policy
        self.requests = requests
        self.metrics = MetricsCollector(config.output_file)

        self.env = simpy.Environment()
        self.fast = StorageTier(self.env, config.fast_tier)
        self.slow = StorageTier(self.env, config.slow_tier)

        self.fast_capacity = simpy.Container(
            self.env,
            capacity=config.fast_tier.capacity,
            init=config.fast_tier.capacity,
        )
        self._origin = 0.0

        # ground-truth filesystem: file_id -> {"fast": size|None, "slow": size|None}
        self._fs: Dict[str, Dict[str, Optional[int]]] = {}
        # per-file fast-tier metadata visible to policy functions
        self._fast_entries: Dict[str, FastEntry] = {}
        # logical file size: max(offset + io_size) seen across all ops
        self._logical_sizes: Dict[str, int] = {}
        # deferred eviction queue: ranked EVICT ops from bg_evict, not yet executed
        self._deferred_evictions: List[Operation] = []

    # -- filesystem helpers -------------------------------------------------

    def _in_tier(self, file_id: str, tier: Tier) -> bool:
        return bool(self._fs.get(file_id, {}).get(tier.value))

    def _file_size_in_tier(self, file_id: str, tier: Tier) -> int:
        return self._fs.get(file_id, {}).get(tier.value) or 0

    def _set_in_tier(self, file_id: str, tier: Tier, size: int) -> None:
        self._fs.setdefault(file_id, {"fast": None, "slow": None})
        self._fs[file_id][tier.value] = size

    def _remove_from_tier(self, file_id: str, tier: Tier) -> int:
        entry = self._fs.get(file_id, {})
        sz = entry.get(tier.value) or 0
        entry[tier.value] = None
        return sz

    def _fs_snapshot(self) -> FsSnapshot:
        return {fid: dict(v) for fid, v in self._fs.items()}

    def _fast_entries_snapshot(self) -> Dict[str, FastEntry]:
        return dict(self._fast_entries)

    # -- SimPy generator processes ------------------------------------------

    def _read_proc(self, op: Operation):
        tier_hw = self.fast if op.tier == Tier.FAST else self.slow
        with tier_hw.read_res.request(priority=op.priority.value) as req:
            yield req
            yield self.env.timeout(tier_hw.read_duration(op.size))
        self.metrics.record_read(op.tier, op.size)
        if op.tier == Tier.SLOW:
            self._set_in_tier(op.file_id, Tier.SLOW, op.size)
            # slow read confirms this version exists; clean only if fast has same size
            if op.file_id in self._fast_entries:
                if op.size == self._file_size_in_tier(op.file_id, Tier.FAST):
                    self._fast_entries[op.file_id].dirty = False
            if self.policy.on_write:
                self.policy.on_write(op.file_id, Tier.SLOW, op.size)

    def _file_size(self, file_id: str) -> int:
        """Current canonical file size (max across tiers)."""
        e = self._fs.get(file_id, {})
        return max(e.get("fast") or 0, e.get("slow") or 0)

    def _write_proc(self, op: Operation, track_stall: bool = False):
        tier_hw = self.fast if op.tier == Tier.FAST else self.slow
        stall_start = self.env.now

        # File occupies its logical size in storage
        logical = self._logical_sizes.get(op.file_id, op.size)
        store_sz = max(logical, op.size)

        if op.tier == Tier.FAST:
            already_fast = self._file_size_in_tier(op.file_id, Tier.FAST)
            needed = max(0, store_sz - already_fast)
            if needed > 0:
                yield self.fast_capacity.get(needed)

        with tier_hw.write_res.request(priority=op.priority.value) as req:
            yield req
            if track_stall:
                self.metrics.global_metrics.total_fast_stall_time += self.env.now - stall_start
            # I/O time based on bytes actually transferred
            yield self.env.timeout(tier_hw.write_duration(op.size))

        if op.tier == Tier.FAST:
            old_fast = self._file_size_in_tier(op.file_id, Tier.FAST)
            self._set_in_tier(op.file_id, Tier.FAST, store_sz)
            if store_sz < old_fast:
                self.fast_capacity.put(old_fast - store_sz)
            self._fast_entries[op.file_id] = FastEntry(size=store_sz, dirty=True)
        else:
            current_slow = self._file_size_in_tier(op.file_id, Tier.SLOW)
            if store_sz >= current_slow:
                self._set_in_tier(op.file_id, Tier.SLOW, store_sz)
            if op.file_id in self._fast_entries:
                if store_sz == self._file_size_in_tier(op.file_id, Tier.FAST):
                    self._fast_entries[op.file_id].dirty = False

        self.metrics.record_write(op.tier, op.size)
        if self.policy.on_write:
            self.policy.on_write(op.file_id, op.tier, store_sz)

    def _evict_proc(self, op: Operation):
        sz = self._file_size_in_tier(op.file_id, Tier.FAST)
        if sz > 0:
            entry = self._fast_entries.get(op.file_id)
            is_dirty = entry.dirty if entry else not self._in_tier(op.file_id, Tier.SLOW)
            if is_dirty and not self.config.always_write_slow:
                # aws=False: no bg slow write guaranteed, must drain dirty files before evicting
                # aws=True: evaluator guarantees a bg slow write is in-flight, safe to skip drain
                drain = Operation(
                    op_type=OpType.WRITE, tier=Tier.SLOW,
                    file_id=op.file_id, size=sz,
                    offset=0, priority=Priority.LOW, primary=False,
                )
                yield self.env.process(self._write_proc(drain))
            self._remove_from_tier(op.file_id, Tier.FAST)
            self.fast_capacity.put(sz)
            self._fast_entries.pop(op.file_id, None)
            if self.policy.on_eviction:
                self.policy.on_eviction(op.file_id, Tier.FAST)
        yield self.env.timeout(0)

    def _migrate_proc(self, op: Operation):
        src_hw = self.fast if op.tier == Tier.FAST else self.slow
        with src_hw.read_res.request(priority=op.priority.value) as req:
            yield req
            yield self.env.timeout(src_hw.read_duration(op.size))
        self.metrics.record_read(op.tier, op.size)
        write_op = Operation(
            op_type=OpType.WRITE, tier=op.dest_tier,
            file_id=op.file_id, size=op.size,
            priority=op.priority, primary=False,
        )
        yield self.env.process(self._write_proc(write_op))

    def _dispatch_op(self, op: Operation, track_fast_stall: bool = False):
        if op.op_type == OpType.READ:
            return self.env.process(self._read_proc(op))
        elif op.op_type == OpType.WRITE:
            return self.env.process(
                self._write_proc(op, track_stall=(op.tier == Tier.FAST and track_fast_stall))
            )
        elif op.op_type == OpType.EVICT:
            return self.env.process(self._evict_proc(op))
        elif op.op_type == OpType.MIGRATE:
            return self.env.process(self._migrate_proc(op))
        raise ValueError(f"Unknown op_type: {op.op_type}")

    # -- fast-full policies -------------------------------------------------

    def _consume_deferred_evictions(self, needed: float, exclude: str = "") -> List[Operation]:
        """Execute deferred evictions just-in-time to free `needed` bytes."""
        evict_ops: List[Operation] = []
        freed = 0.0
        remaining = []
        for op in self._deferred_evictions:
            if freed >= needed:
                remaining.append(op)
                continue
            if op.file_id == exclude:
                remaining.append(op)
                continue
            if not self._in_tier(op.file_id, Tier.FAST):
                continue
            sz = self._file_size_in_tier(op.file_id, Tier.FAST)
            evict_ops.append(Operation(
                OpType.EVICT, Tier.FAST, op.file_id, sz, primary=False))
            freed += sz
        self._deferred_evictions = remaining
        return evict_ops

    def _apply_fast_full_policy(self, ops: List[Operation]) -> List[Operation]:
        if self.config.on_fast_full != "spill":
            return ops
        result: List[Operation] = []
        for op in ops:
            if op.op_type == OpType.WRITE and op.tier == Tier.FAST:
                logical = self._logical_sizes.get(op.file_id, op.size)
                store_sz = max(logical, op.size)
                already = self._file_size_in_tier(op.file_id, Tier.FAST)
                needed = max(0, store_sz - already)
                if needed > 0 and self.fast_capacity.level < needed:
                    evict_ops = self._consume_deferred_evictions(
                        needed - self.fast_capacity.level, exclude=op.file_id)
                    if evict_ops:
                        result.extend(evict_ops)
                        result.append(op)
                        continue
                    result.append(Operation(
                        op_type=OpType.WRITE, tier=Tier.SLOW,
                        file_id=op.file_id, size=op.size,
                        offset=op.offset, priority=op.priority,
                        primary=op.primary,
                    ))
                    continue
            result.append(op)
        return result

    def _maybe_auto_write_slow(self, request: Request, ops: List[Operation]):
        if not self.config.always_write_slow:
            return
        has_slow = any(o.op_type == OpType.WRITE and o.tier == Tier.SLOW for o in ops)
        if not has_slow:
            ops.append(Operation(
                op_type=OpType.WRITE, tier=Tier.SLOW,
                file_id=request.file_id, size=request.size,
                offset=request.offset, priority=Priority.LOW, primary=False,
            ))  # _write_proc handles append sizing

    # -- recall: read from slow before writing to fast ----------------------

    def _insert_recall_if_needed(self, ops: List[Operation]) -> List[Operation]:
        """If a WRITE FAST targets a file only in slow, prepend a READ SLOW to recall it."""
        result: List[Operation] = []
        for op in ops:
            if (op.op_type == OpType.WRITE and op.tier == Tier.FAST
                    and not self._in_tier(op.file_id, Tier.FAST)
                    and self._in_tier(op.file_id, Tier.SLOW)):
                slow_sz = self._file_size_in_tier(op.file_id, Tier.SLOW)
                recall = Operation(
                    op_type=OpType.READ, tier=Tier.SLOW,
                    file_id=op.file_id, size=slow_sz,
                    priority=op.priority, primary=op.primary,
                )
                result.append(recall)
            result.append(op)
        return result

    # -- auto-eviction before a WRITE FAST op (non-spill mode) --------------

    def _prepend_evictions_if_needed(self, ops: List[Operation]) -> List[Operation]:
        if self.config.on_fast_full == "spill":
            return ops
        extra: List[Operation] = []
        for op in ops:
            if op.primary and op.op_type == OpType.WRITE and op.tier == Tier.FAST:
                logical = self._logical_sizes.get(op.file_id, op.size)
                store_sz = max(logical, op.size)
                already = self._file_size_in_tier(op.file_id, Tier.FAST)
                needed = max(0, store_sz - already)
                shortfall = max(0.0, needed - self.fast_capacity.level)
                if shortfall > 0:
                    deferred = self._consume_deferred_evictions(
                        shortfall, exclude=op.file_id)
                    extra.extend(deferred)
                    deferred_freed = sum(
                        self._file_size_in_tier(e.file_id, Tier.FAST)
                        for e in deferred)
                    remaining_shortfall = shortfall - deferred_freed
                    if remaining_shortfall > 0:
                        evict_ops = self.policy.evict_bytes(
                            remaining_shortfall,
                            self._fast_entries_snapshot(),
                            self.fast_capacity.level,
                            op.file_id,
                        )
                        extra.extend(evict_ops)
        return extra + ops

    # -- per-request process ------------------------------------------------

    def _handle_request(self, request: Request):
        remaining = self.fast_capacity.level
        in_fast = self._in_tier(request.file_id, Tier.FAST)
        in_slow = self._in_tier(request.file_id, Tier.SLOW)
        fe_snap = self._fast_entries_snapshot()
        fs_snap = self._fs_snapshot()

        fs_snap["_arrival_time"] = request.arrival_time

        # Track logical file size: max(offset + io_size) seen
        extent_end = request.offset + request.size
        prev_logical = self._logical_sizes.get(request.file_id, 0)
        self._logical_sizes[request.file_id] = max(prev_logical, extent_end)

        t0 = wall_clock.perf_counter()
        is_new_write = request.op_type == OpType.WRITE and not in_fast and not in_slow
        if is_new_write:
            ops = self.policy.place_new_write(
                request.file_id, self._logical_sizes[request.file_id], request.offset,
                remaining, fe_snap, fs_snap,
            )
        else:
            ops = self.policy.handle_existing(
                request, in_fast, in_slow, remaining, fe_snap, fs_snap,
            )
        decision_time = wall_clock.perf_counter() - t0

        ops = self._insert_recall_if_needed(ops)
        ops = self._prepend_evictions_if_needed(ops)
        ops = self._apply_fast_full_policy(ops)

        if request.op_type == OpType.WRITE:
            self.metrics.record_requested_write(request.size)
            self._maybe_auto_write_slow(request, ops)

        primary_ops = [o for o in ops if o.primary]
        bg_ops = [o for o in ops if not o.primary]

        hit = (
            request.op_type == OpType.READ
            and any(o.op_type == OpType.READ and o.tier == Tier.FAST for o in primary_ops)
        )

        if self.config.model_decision_time and decision_time > 0:
            yield self.env.timeout(decision_time)

        for op in bg_ops:
            self._dispatch_op(op)

        primary_events = [self._dispatch_op(op, track_fast_stall=True) for op in primary_ops]
        yield simpy.AllOf(self.env, primary_events)

        completion_time = self.env.now
        served = "fast" if hit else "slow"
        if request.op_type == OpType.WRITE:
            pw = next((o for o in primary_ops if o.op_type == OpType.WRITE), None)
            served = pw.tier.value if pw else "slow"

        norm_arrival = request.arrival_time - self._origin
        self.metrics.record_request(RequestRecord(
            request_type=request.op_type.name,
            request_size=request.size,
            served_tier=served,
            arrival_time=norm_arrival,
            completion_time=completion_time,
            hit=hit,
            is_warmup=request.is_warmup,
            dpa_decision_time_s=decision_time,
        ))

    # -- trace replayer and background processes ----------------------------

    def _trace_replayer(self):
        requests = sorted(self.requests, key=lambda r: r.arrival_time)
        if not requests:
            return
        origin = requests[0].arrival_time
        self._origin = origin
        for req in requests:
            wait = req.arrival_time - origin - self.env.now
            if wait > 0:
                yield self.env.timeout(wait)
            self.env.process(self._handle_request(req))

    def _background_dpa(self):
        interval = self.policy.bg_interval
        while True:
            yield self.env.timeout(interval)
            sim_time = self.env.now
            self.metrics.record_background_activation(sim_time)
            fe = self._fast_entries_snapshot()
            free = self.fast_capacity.level
            evict_ops = self.policy.bg_evict(fe, free, sim_time)
            if evict_ops:
                self._deferred_evictions = evict_ops
            for op in self.policy.bg_promote(fe, free, sim_time):
                self._dispatch_op(op)

    # -- public API ---------------------------------------------------------

    def run(self) -> MetricsCollector:
        if not self.requests:
            self.metrics.flush()
            return self.metrics

        last_ts = max(r.arrival_time for r in self.requests)
        first_ts = min(r.arrival_time for r in self.requests)
        max_size = max(r.size for r in self.requests)
        min_bw = min(self.config.slow_tier.write_bandwidth,
                     self.config.fast_tier.write_bandwidth)
        drain_buffer = max_size / min_bw + self.config.slow_tier.write_latency + 10.0
        sim_end = (last_ts - first_ts) + drain_buffer

        self.env.process(self._trace_replayer())
        self.env.process(self._background_dpa())
        self.env.run(until=sim_end)
        self.metrics.flush()
        return self.metrics
