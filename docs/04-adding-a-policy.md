# Adding Your Own Policy

To create a custom eviction/placement policy, you implement 5 functions and bundle them into a `PolicyFunctions` dataclass.

## The Interface

```python
from src.policy import PolicyFunctions
from src.models import FastEntry, Operation, OpType, Request, Tier

PolicyFunctions(
    place_new_write=...,    # where to put new files
    handle_existing=...,    # how to serve known files
    evict_bytes=...,        # which files to evict on demand
    bg_evict=...,           # proactive background eviction
    bg_promote=...,         # background promotion from slow to fast
    bg_interval=5.0,        # seconds between background cycles
    on_eviction=...,        # callback: file removed from tier (optional)
    on_write=...,           # callback: write completed (optional)
)
```

## Function Signatures

### 1. place_new_write

Called when a WRITE arrives for a file not yet in either tier.

```python
def place_new_write(
    file_id: str,           # the file being written
    size: int,              # logical file size (max offset+size seen)
    offset: int,            # byte offset of this write
    free: float,            # bytes available in fast tier
    fast_entries: dict,     # {fid: FastEntry(size, dirty)} snapshot
    fs_snapshot: dict,      # {fid: {"fast": size|None, "slow": size|None}}
) -> List[Operation]:
```

**Typical return**: one `Operation(OpType.WRITE, Tier.FAST, file_id, size)` or `Tier.SLOW`.

### 2. handle_existing

Called for any request (READ or WRITE) targeting a file already known to the system.

```python
def handle_existing(
    request: Request,       # the incoming request (has .op_type, .file_id, .size, .offset)
    in_fast: bool,          # file is currently in fast tier
    in_slow: bool,          # file is currently in slow tier
    free: float,            # bytes available in fast tier
    fast_entries: dict,     # snapshot
    fs_snapshot: dict,      # snapshot
) -> List[Operation]:
```

**Typical returns**:
- READ from fast: `[Operation(OpType.READ, Tier.FAST, fid, size)]`
- READ from slow (miss): `[Operation(OpType.READ, Tier.SLOW, fid, size)]`
- WRITE to fast: `[Operation(OpType.WRITE, Tier.FAST, fid, size)]`
- Promote on read: `[Operation(OpType.READ, Tier.SLOW, fid, size), Operation(OpType.WRITE, Tier.FAST, fid, size, primary=False)]`

### 3. evict_bytes

Called when the fast tier is full and a write needs space. Must return EVICT operations freeing at least `needed` bytes.

```python
def evict_bytes(
    needed: float,          # bytes that must be freed
    fast_entries: dict,     # current fast tier contents
    free: float,            # current free space
    writing_fid: str,       # file being written (avoid evicting it)
) -> List[Operation]:
```

**Return**: list of `Operation(OpType.EVICT, Tier.FAST, victim_fid, victim_size)`.

### 4. bg_evict

Called periodically. Return eviction operations to proactively free space. These are queued and consumed on-demand (not executed immediately).

```python
def bg_evict(
    fast_entries: dict,     # current fast tier contents
    free: float,            # current free space
    sim_time: float,        # current simulation timestamp
) -> List[Operation]:
```

### 5. bg_promote

Called periodically. Return write operations to promote hot files from slow to fast.

```python
def bg_promote(
    fast_entries: dict,
    free: float,
    sim_time: float,
) -> List[Operation]:
```

## Minimal Example: Random Eviction Policy

```python
import random
from src.policy import PolicyFunctions
from src.models import FastEntry, Operation, OpType, Request, Tier


def make_random_policy(fast_capacity: float) -> PolicyFunctions:
    """Evicts random files when space is needed."""

    def place_new_write(fid, size, offset, free, fast_entries, fs):
        return [Operation(OpType.WRITE, Tier.FAST, fid, size)]

    def handle_existing(request, in_fast, in_slow, free, fast_entries, fs):
        tier = Tier.FAST if in_fast else Tier.SLOW
        return [Operation(request.op_type, tier, request.file_id, request.size)]

    def evict_bytes(needed, fast_entries, free, writing_fid):
        ops = []
        freed = 0
        candidates = [
            (fid, entry) for fid, entry in fast_entries.items()
            if fid != writing_fid
        ]
        random.shuffle(candidates)
        for fid, entry in candidates:
            if freed >= needed:
                break
            ops.append(Operation(OpType.EVICT, Tier.FAST, fid, entry.size))
            freed += entry.size
        return ops

    def bg_evict(fast_entries, free, sim_time):
        return []  # no proactive eviction

    def bg_promote(fast_entries, free, sim_time):
        return []  # no promotion

    return PolicyFunctions(
        place_new_write=place_new_write,
        handle_existing=handle_existing,
        evict_bytes=evict_bytes,
        bg_evict=bg_evict,
        bg_promote=bg_promote,
    )
```

## Using Your Policy

```python
from src.config import TierConfig, EvaluatorConfig
from src.trace_loader import load_thesios_csv
from src.evaluator import Evaluator

# your policy
policy = make_random_policy(fast_capacity=10 * 1024**3)

config = EvaluatorConfig(
    fast_tier=TierConfig(
        read_bandwidth=3.5e9, write_bandwidth=3.0e9,
        read_latency=0.00002, write_latency=0.00002,
        max_concurrent_reads=32, max_concurrent_writes=32,
        capacity=10 * 1024**3,
    ),
    slow_tier=TierConfig(
        read_bandwidth=200e6, write_bandwidth=150e6,
        read_latency=0.005, write_latency=0.005,
        max_concurrent_reads=8, max_concurrent_writes=8,
    ),
    trace_file="traces/workload2/cluster_1_group_1",
    output_file="random_policy_results.csv",
    warmup_ops=71000,
    always_write_slow=True,
)

requests = load_thesios_csv(config.trace_file, max_rows=142000, warmup_rows=71000)
evaluator = Evaluator(config, policy, requests)
metrics = evaluator.run()

print(f"Hit rate: {metrics.global_metrics.hit_rate:.4f}")
```

## Tracking Internal State

Most policies need to track access history. Use the optional callbacks:

```python
def make_my_policy(fast_capacity: float) -> PolicyFunctions:
    access_times = {}  # fid -> last access time
    tracked_files = {} # fid -> size in fast

    def on_write(fid, tier, size):
        """Called after every successful write."""
        if tier == Tier.FAST:
            tracked_files[fid] = size

    def on_eviction(fid, tier):
        """Called after a file is removed from a tier."""
        if tier == Tier.FAST:
            tracked_files.pop(fid, None)

    # ... define the 5 functions using access_times / tracked_files ...

    return PolicyFunctions(
        place_new_write=place_new_write,
        handle_existing=handle_existing,
        evict_bytes=evict_bytes,
        bg_evict=bg_evict,
        bg_promote=bg_promote,
        on_eviction=on_eviction,
        on_write=on_write,
    )
```

The evaluator calls `on_write` after every completed write and `on_eviction` after every completed eviction, so your internal state stays synchronized with the simulator's filesystem.

## Tips

- **`fast_entries` is a snapshot** (copy). It shows you what's in the fast tier right now, but modifying it doesn't affect the simulator.
- **`fs_snapshot["_arrival_time"]`** gives you the current request's timestamp, useful for recency calculations.
- **Return empty lists** from `bg_evict`/`bg_promote` if you don't need background maintenance.
- **The `writing_fid` parameter** in `evict_bytes` tells you which file is waiting for space. Don't evict it (you'd evict what you're about to write).
- **`Operation.primary = False`** means the evaluator won't wait for it. Use this for background copies (e.g., promoting a file to fast after serving a read from slow).
- **Evict operations must target files in `fast_entries`**. Evicting a file not in fast tier is a no-op.
