from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import OrderedDict, defaultdict
from typing import Dict, List, Set

from .models import Request, Operation, OpType, Tier, Priority


@dataclass
class DPAConfig:
    background_interval: float = 10.0


class DataPlacementAlgorithm(ABC):
    @abstractmethod
    def get_config(self) -> DPAConfig: ...

    @abstractmethod
    def on_request(self, request: Request, remaining_fast_bytes: float) -> List[Operation]:
        """Return ops for the evaluator to execute. Exactly one primary=True op expected."""
        ...

    @abstractmethod
    def on_completion(self, request: Request) -> List[Operation]: ...

    @abstractmethod
    def run_background(self, sim_time: float) -> List[Operation]: ...

    def notify_eviction(self, file_id: str, tier: Tier) -> None:
        pass

    def notify_write(self, file_id: str, tier: Tier, size: int) -> None:
        pass


# ---------------------------------------------------------------------------
# Shared base for caching-style policies
# ---------------------------------------------------------------------------

class _BaseCachingPolicy(DataPlacementAlgorithm):
    def __init__(
        self,
        fast_capacity: float,
        promote_on_miss: bool = True,
        eviction_headroom: float = 0.05,
        max_file_size_pct: float = 1.0,
        fast_fill_threshold: float = 1.0,
        background_interval: float = 5.0,
        write_behind: bool = False,
    ):
        self.fast_capacity = fast_capacity
        self.promote_on_miss = promote_on_miss
        self.eviction_headroom = eviction_headroom
        self.max_file_size_pct = max_file_size_pct
        self.fast_fill_threshold = fast_fill_threshold
        self._background_interval = background_interval
        # proactively write fast-only files to slow in eviction order each bg cycle
        self.write_behind = write_behind
        self._fast_used: float = 0.0
        self._on_slow: Set[str] = set()

    def get_config(self) -> DPAConfig:
        return DPAConfig(background_interval=self._background_interval)

    def _too_large(self, size: int) -> bool:
        return size > self.max_file_size_pct * self.fast_capacity

    def _eviction_target(self, needed: float) -> float:
        return needed + self.eviction_headroom * self.fast_capacity

    def on_completion(self, request: Request) -> List[Operation]:
        return []

    def notify_write(self, file_id: str, tier: Tier, size: int) -> None:
        if tier == Tier.SLOW:
            self._on_slow.add(file_id)


# ---------------------------------------------------------------------------
# LRU
# ---------------------------------------------------------------------------

class LRUPolicy(_BaseCachingPolicy):
    """
    Fast tier as LRU cache.
    Writes go to fast first; slow write added by evaluator (always_write_slow).
    Background: evicts LRU tail when fast_used > fast_fill_threshold.
    """

    def __init__(self, fast_capacity: float, promote_on_miss: bool = True,
                 eviction_headroom: float = 0.05, max_file_size_pct: float = 1.0,
                 fast_fill_threshold: float = 1.0, background_interval: float = 30.0,
                 write_behind: bool = False):
        super().__init__(fast_capacity, promote_on_miss, eviction_headroom,
                         max_file_size_pct, fast_fill_threshold, background_interval,
                         write_behind)
        self._lru: OrderedDict[str, int] = OrderedDict()

    def _touch(self, file_id: str, size: int) -> None:
        if file_id in self._lru:
            old = self._lru[file_id]
            self._fast_used += size - old
            self._lru[file_id] = size
            self._lru.move_to_end(file_id)
        else:
            self._lru[file_id] = size
            self._fast_used += size

    def _pick_evictions(self, needed: float) -> List[Operation]:
        target = self._eviction_target(needed)
        ops: List[Operation] = []
        freed = 0.0
        for fid, sz in list(self._lru.items()):
            if freed >= target:
                break
            ops.append(Operation(OpType.EVICT, Tier.FAST, fid, sz, primary=False))
            freed += sz
        return ops

    def on_request(self, request: Request, remaining: float) -> List[Operation]:
        ops: List[Operation] = []
        fid, sz = request.file_id, request.size
        in_fast = fid in self._lru

        if self._too_large(sz):
            ops.append(Operation(request.op_type, Tier.SLOW, fid, sz,
                                 offset=request.offset, primary=True))
            return ops

        if request.op_type == OpType.READ:
            if in_fast:
                self._touch(fid, sz)
                ops.append(Operation(OpType.READ, Tier.FAST, fid, sz,
                                     offset=request.offset, primary=True))
            else:
                ops.append(Operation(OpType.READ, Tier.SLOW, fid, sz,
                                     offset=request.offset, primary=True))
                if self.promote_on_miss:
                    if remaining < sz:
                        ops.extend(self._pick_evictions(sz - remaining))
                    ops.append(Operation(OpType.WRITE, Tier.FAST, fid, sz,
                                         offset=request.offset, primary=False))
                    self._touch(fid, sz)

        elif request.op_type == OpType.WRITE:
            if in_fast or remaining >= sz:
                if not in_fast and remaining < sz:
                    ops.extend(self._pick_evictions(sz - remaining))
                self._touch(fid, sz)
                ops.append(Operation(OpType.WRITE, Tier.FAST, fid, sz,
                                     offset=request.offset, primary=True))
            else:
                ops.append(Operation(OpType.WRITE, Tier.SLOW, fid, sz,
                                     offset=request.offset, primary=True))

        return ops

    def _pick_write_behind(self) -> List[Operation]:
        """WRITE SLOW ops for fast-only files in LRU coldest-first order."""
        ops: List[Operation] = []
        for fid, sz in self._lru.items():
            if fid not in self._on_slow:
                ops.append(Operation(OpType.WRITE, Tier.SLOW, fid, sz, primary=False))
        return ops

    def run_background(self, sim_time: float) -> List[Operation]:
        ops: List[Operation] = []
        if self._fast_used > self.fast_fill_threshold * self.fast_capacity:
            excess = self._fast_used - self.fast_fill_threshold * self.fast_capacity
            ops.extend(self._pick_evictions(excess))
        if self.write_behind:
            ops.extend(self._pick_write_behind())
        return ops

    def notify_eviction(self, file_id: str, tier: Tier) -> None:
        if tier == Tier.FAST and file_id in self._lru:
            self._fast_used -= self._lru.pop(file_id)

    def notify_write(self, file_id: str, tier: Tier, size: int) -> None:
        super().notify_write(file_id, tier, size)
        if tier == Tier.FAST:
            self._touch(file_id, size)


# ---------------------------------------------------------------------------
# Size-Aware LRU
# ---------------------------------------------------------------------------

class SizeAwareLRUPolicy(LRUPolicy):
    """
    Like LRU but evicts largest files within the cold tail first.
    Trades recency precision for lower write amplification (fewer evictions needed).
    """

    def __init__(self, fast_capacity: float, promote_on_miss: bool = True,
                 eviction_headroom: float = 0.05, max_file_size_pct: float = 1.0,
                 fast_fill_threshold: float = 1.0, background_interval: float = 30.0,
                 lru_window: float = 0.5):
        super().__init__(fast_capacity, promote_on_miss, eviction_headroom,
                         max_file_size_pct, fast_fill_threshold, background_interval)
        # fraction of LRU tail considered eviction candidates
        self.lru_window = lru_window

    def _pick_evictions(self, needed: float) -> List[Operation]:
        items = list(self._lru.items())
        window_n = max(1, int(len(items) * self.lru_window))
        # coldest items sorted by size descending
        candidates = sorted(items[:window_n], key=lambda x: x[1], reverse=True)
        target = self._eviction_target(needed)
        ops: List[Operation] = []
        freed = 0.0
        for fid, sz in candidates:
            if freed >= target:
                break
            ops.append(Operation(OpType.EVICT, Tier.FAST, fid, sz, primary=False))
            freed += sz
        return ops


# ---------------------------------------------------------------------------
# LFU
# ---------------------------------------------------------------------------

class LFUPolicy(_BaseCachingPolicy):
    """
    Frequency-based cache. Promotes slow-tier items when their access frequency
    exceeds the least-frequent fast-tier item by `promotion_threshold` (default 1.25x).
    Writes behave like LRU (fast-first if space allows).
    """

    def __init__(self, fast_capacity: float, promote_on_miss: bool = True,
                 eviction_headroom: float = 0.05, max_file_size_pct: float = 1.0,
                 fast_fill_threshold: float = 1.0, background_interval: float = 30.0,
                 write_behind: bool = False, promotion_threshold: float = 0.5):
        super().__init__(fast_capacity, promote_on_miss, eviction_headroom,
                         max_file_size_pct, fast_fill_threshold, background_interval,
                         write_behind)
        self.promotion_threshold = promotion_threshold
        self._freq: Dict[str, int] = defaultdict(int)
        self._in_fast: Dict[str, int] = {}  # file_id -> size

    def _min_fast_freq(self) -> int:
        if not self._in_fast:
            return 0
        return min(self._freq[fid] for fid in self._in_fast)

    def _pick_evictions(self, needed: float) -> List[Operation]:
        # evict by lowest frequency; break ties by largest size
        candidates = sorted(self._in_fast.items(),
                             key=lambda x: (self._freq[x[0]], -x[1]))
        target = self._eviction_target(needed)
        ops: List[Operation] = []
        freed = 0.0
        for fid, sz in candidates:
            if freed >= target:
                break
            ops.append(Operation(OpType.EVICT, Tier.FAST, fid, sz, primary=False))
            freed += sz
        return ops

    def on_request(self, request: Request, remaining: float) -> List[Operation]:
        ops: List[Operation] = []
        fid, sz = request.file_id, request.size
        in_fast = fid in self._in_fast
        self._freq[fid] += 1

        if self._too_large(sz):
            ops.append(Operation(request.op_type, Tier.SLOW, fid, sz,
                                 offset=request.offset, primary=True))
            return ops

        if request.op_type == OpType.READ:
            if in_fast:
                ops.append(Operation(OpType.READ, Tier.FAST, fid, sz,
                                     offset=request.offset, primary=True))
            else:
                ops.append(Operation(OpType.READ, Tier.SLOW, fid, sz,
                                     offset=request.offset, primary=True))
                if self.promote_on_miss:
                    min_freq = self._min_fast_freq()
                    should_promote = (min_freq == 0 or
                                      self._freq[fid] > self.promotion_threshold * min_freq)
                    if should_promote:
                        if remaining < sz:
                            ops.extend(self._pick_evictions(sz - remaining))
                        ops.append(Operation(OpType.WRITE, Tier.FAST, fid, sz,
                                             offset=request.offset, primary=False))
                        self._in_fast[fid] = sz
                        self._fast_used += sz

        elif request.op_type == OpType.WRITE:
            if in_fast or remaining >= sz:
                if not in_fast and remaining < sz:
                    ops.extend(self._pick_evictions(sz - remaining))
                if not in_fast:
                    self._in_fast[fid] = sz
                    self._fast_used += sz
                ops.append(Operation(OpType.WRITE, Tier.FAST, fid, sz,
                                     offset=request.offset, primary=True))
            else:
                ops.append(Operation(OpType.WRITE, Tier.SLOW, fid, sz,
                                     offset=request.offset, primary=True))

        return ops

    def _pick_write_behind(self) -> List[Operation]:
        """WRITE SLOW ops for fast-only files in LFU eviction order (lowest freq first)."""
        candidates = sorted(self._in_fast.items(),
                            key=lambda x: (self._freq[x[0]], -x[1]))
        return [
            Operation(OpType.WRITE, Tier.SLOW, fid, sz, primary=False)
            for fid, sz in candidates if fid not in self._on_slow
        ]

    def run_background(self, sim_time: float) -> List[Operation]:
        ops: List[Operation] = []
        if self._fast_used > self.fast_fill_threshold * self.fast_capacity:
            excess = self._fast_used - self.fast_fill_threshold * self.fast_capacity
            ops.extend(self._pick_evictions(excess))
        if self.write_behind:
            ops.extend(self._pick_write_behind())
        return ops

    def notify_eviction(self, file_id: str, tier: Tier) -> None:
        if tier == Tier.FAST and file_id in self._in_fast:
            self._fast_used -= self._in_fast.pop(file_id)

    def notify_write(self, file_id: str, tier: Tier, size: int) -> None:
        super().notify_write(file_id, tier, size)
        if tier == Tier.FAST:
            old = self._in_fast.get(file_id, 0)
            self._fast_used += size - old
            self._in_fast[file_id] = size


# ---------------------------------------------------------------------------
# S3FIFO (modified for tiered storage)
# ---------------------------------------------------------------------------

class S3FIFOPolicy(_BaseCachingPolicy):
    """
    S3FIFO variant adapted for tiered storage.

    Cold-start mode (cold_start_to_main=True, default): while main queue is not yet full,
      new writes go directly to main queue (LRU). Once main is full, switch to normal mode.
    Normal mode:
      Small queue (small_ratio of fast capacity): new entries, FIFO eviction.
        - freq > 1 on eviction -> promoted to main queue (stays in fast, no I/O)
        - freq <= 1 -> physical eviction + ghost.add(fid)
      Main queue (1 - small_ratio): LRU eviction.
    Ghost set (in-memory only): populated by small-queue evictions AND first cold read misses.
      - Second cold read of a ghost item -> promote to main queue.
    Background: keep small queue >= 5% free; enforce fill threshold.
    """

    SMALL_MIN_FREE = 0.05

    def __init__(self, fast_capacity: float, promote_on_miss: bool = True,
                 eviction_headroom: float = 0.05, max_file_size_pct: float = 1.0,
                 fast_fill_threshold: float = 1.0, background_interval: float = 10.0,
                 write_behind: bool = False,
                 small_ratio: float = 0.10, cold_start_to_main: bool = True):
        super().__init__(fast_capacity, promote_on_miss, eviction_headroom,
                         max_file_size_pct, fast_fill_threshold, background_interval,
                         write_behind)
        self.small_ratio = small_ratio
        self.cold_start_to_main = cold_start_to_main
        self._small_cap = fast_capacity * small_ratio
        self._main_cap = fast_capacity * (1 - small_ratio)

        self._small_q: OrderedDict[str, int] = OrderedDict()  # FIFO
        self._small_used: float = 0.0
        self._main_q: OrderedDict[str, int] = OrderedDict()   # LRU
        self._main_used: float = 0.0

        self._freq: Dict[str, int] = defaultdict(int)
        self._ghost: Set[str] = set()

    def _in_fast(self, fid: str) -> bool:
        return fid in self._small_q or fid in self._main_q

    def _in_cold_fill(self) -> bool:
        return self.cold_start_to_main and self._main_used < self._main_cap

    # -- internal eviction helpers (modify DPA state eagerly) ---------------

    def _drain_small(self, needed: float) -> List[Operation]:
        """Free `needed` bytes from small queue; hot items move to main (no I/O)."""
        ops: List[Operation] = []
        freed = 0.0
        while freed < needed and self._small_q:
            fid, sz = next(iter(self._small_q.items()))
            self._small_q.pop(fid)
            self._small_used -= sz
            freed += sz
            if self._freq.get(fid, 0) > 1:
                main_need = max(0.0, sz - (self._main_cap - self._main_used))
                if main_need > 0:
                    ops.extend(self._drain_main(main_need))
                self._main_q[fid] = sz
                self._main_used += sz
            else:
                self._ghost.add(fid)
                ops.append(Operation(OpType.EVICT, Tier.FAST, fid, sz, primary=False))
        return ops

    def _drain_main(self, needed: float) -> List[Operation]:
        """Evict LRU from main queue."""
        ops: List[Operation] = []
        freed = 0.0
        while freed < needed and self._main_q:
            fid, sz = next(iter(self._main_q.items()))
            self._main_q.pop(fid)
            self._main_used -= sz
            ops.append(Operation(OpType.EVICT, Tier.FAST, fid, sz, primary=False))
            freed += sz
        return ops

    def on_request(self, request: Request, remaining: float) -> List[Operation]:
        ops: List[Operation] = []
        fid, sz = request.file_id, request.size
        self._freq[fid] += 1

        if self._too_large(sz):
            ops.append(Operation(request.op_type, Tier.SLOW, fid, sz,
                                 offset=request.offset, primary=True))
            return ops

        if request.op_type == OpType.READ:
            if fid in self._small_q or fid in self._main_q:
                if fid in self._main_q:
                    self._main_q.move_to_end(fid)
                ops.append(Operation(OpType.READ, Tier.FAST, fid, sz,
                                     offset=request.offset, primary=True))
            else:
                ops.append(Operation(OpType.READ, Tier.SLOW, fid, sz,
                                     offset=request.offset, primary=True))
                if self.promote_on_miss:
                    if fid in self._ghost or self._in_cold_fill():
                        self._ghost.discard(fid)
                        main_need = max(0.0, sz - (self._main_cap - self._main_used))
                        if main_need > 0:
                            ops.extend(self._drain_main(main_need))
                        ops.append(Operation(OpType.WRITE, Tier.FAST, fid, sz,
                                             offset=request.offset, primary=False))
                        self._main_q[fid] = sz
                        self._main_used += sz
                    else:
                        self._ghost.add(fid)

        elif request.op_type == OpType.WRITE:
            if fid in self._small_q:
                old = self._small_q[fid]
                self._small_used += sz - old
                self._small_q[fid] = sz
                ops.append(Operation(OpType.WRITE, Tier.FAST, fid, sz,
                                     offset=request.offset, primary=True))
            elif fid in self._main_q:
                old = self._main_q[fid]
                self._main_used += sz - old
                self._main_q[fid] = sz
                self._main_q.move_to_end(fid)
                ops.append(Operation(OpType.WRITE, Tier.FAST, fid, sz,
                                     offset=request.offset, primary=True))
            elif self._in_cold_fill() and remaining >= sz:
                # cold-start: fill main queue directly (skip small queue filter)
                needed = max(0.0, sz - (self._main_cap - self._main_used))
                if needed > 0:
                    ops.extend(self._drain_main(needed))
                ops.append(Operation(OpType.WRITE, Tier.FAST, fid, sz,
                                     offset=request.offset, primary=True))
                self._main_q[fid] = sz
                self._main_q.move_to_end(fid)
                self._main_used += sz
                self._ghost.discard(fid)
            elif remaining >= sz:
                # normal: new entry goes into small queue
                ops.extend(self._drain_small(max(0, sz - (self._small_cap - self._small_used))))
                ops.append(Operation(OpType.WRITE, Tier.FAST, fid, sz,
                                     offset=request.offset, primary=True))
                self._small_q[fid] = sz
                self._small_used += sz
                self._ghost.discard(fid)
            else:
                ops.append(Operation(OpType.WRITE, Tier.SLOW, fid, sz,
                                     offset=request.offset, primary=True))

        self._fast_used = self._small_used + self._main_used
        return ops

    def _pick_write_behind(self) -> List[Operation]:
        """WRITE SLOW ops in eviction order: small queue (FIFO) then main queue (LRU coldest)."""
        ops: List[Operation] = []
        for fid, sz in self._small_q.items():
            if fid not in self._on_slow:
                ops.append(Operation(OpType.WRITE, Tier.SLOW, fid, sz, primary=False))
        for fid, sz in self._main_q.items():
            if fid not in self._on_slow:
                ops.append(Operation(OpType.WRITE, Tier.SLOW, fid, sz, primary=False))
        return ops

    def run_background(self, sim_time: float) -> List[Operation]:
        ops: List[Operation] = []
        min_free = self._small_cap * self.SMALL_MIN_FREE
        if self._small_used > self._small_cap - min_free:
            ops.extend(self._drain_small(self._small_used - (self._small_cap - min_free)))
        total = self._small_used + self._main_used
        if total > self.fast_fill_threshold * self.fast_capacity:
            ops.extend(self._drain_main(total - self.fast_fill_threshold * self.fast_capacity))
        if self.write_behind:
            ops.extend(self._pick_write_behind())
        self._fast_used = self._small_used + self._main_used
        return ops

    def notify_eviction(self, file_id: str, tier: Tier) -> None:
        # state already updated eagerly; handle any external evictions
        if tier == Tier.FAST:
            if file_id in self._small_q:
                self._small_used -= self._small_q.pop(file_id)
            elif file_id in self._main_q:
                self._main_used -= self._main_q.pop(file_id)
            self._fast_used = self._small_used + self._main_used

    def notify_write(self, file_id: str, tier: Tier, size: int) -> None:
        super().notify_write(file_id, tier, size)
        if tier == Tier.FAST:
            if file_id in self._small_q:
                old = self._small_q[file_id]
                self._small_used += size - old
                self._small_q[file_id] = size
            elif file_id in self._main_q:
                old = self._main_q[file_id]
                self._main_used += size - old
                self._main_q[file_id] = size
            self._fast_used = self._small_used + self._main_used
