# How the Evaluator Works

## Simulation Loop

The evaluator replays a trace of I/O requests through SimPy's event-driven scheduler:

```
1. Load trace --> sorted List[Request] by arrival_time
2. Normalize timestamps (first request = t=0)
3. For each request:
   a. Wait until its arrival time
   b. Determine file state (in fast? in slow? new?)
   c. Call the appropriate policy function
   d. Execute resulting I/O operations with realistic timing
   e. Record metrics
4. Background process runs every bg_interval seconds:
   - Calls policy.bg_evict() --> queues deferred evictions
   - Calls policy.bg_promote() --> dispatches promotions immediately
5. Simulation ends after last request + drain buffer
6. Flush metrics to CSV
```

## Request Handling Detail

When a request arrives, the evaluator follows this path:

### New file (first WRITE)
```
policy.place_new_write(file_id, size, offset, free_space, fast_entries, fs_snapshot)
  --> returns List[Operation] (typically one WRITE to fast or slow)
```

### Existing file (any READ or WRITE)
```
policy.handle_existing(request, in_fast, in_slow, free_space, fast_entries, fs_snapshot)
  --> returns List[Operation]
```

### Eviction (fast tier full, non-spill mode)
```
1. Check if deferred evictions from bg_evict cover the needed space
2. If not: policy.evict_bytes(needed_bytes, fast_entries, free, writing_fid)
   --> returns List[Operation] with EVICT ops totaling >= needed bytes
3. Execute evictions before the write proceeds
```

## I/O Timing Model

Each operation goes through:

1. **Resource acquisition**: wait for a read/write slot (priority queue)
2. **Duration**: `latency + size / bandwidth`
3. **State update**: filesystem metadata updated after completion

Operations are classified as:
- **Primary** (`op.primary = True`): the request waits for these to complete
- **Background** (`op.primary = False`): fire-and-forget, runs concurrently

## Capacity Management

The fast tier's free space is tracked with a SimPy `Container`:

- **WRITE FAST**: decreases available capacity by `max(logical_size, op.size)`
- **EVICT**: increases available capacity by the stored file size
- **On full** (configurable):
  - `"wait"`: block until evictions free space (default behavior with evict_bytes)
  - `"spill"`: redirect the write to slow tier immediately

## Always-Write-Slow Mode

When `always_write_slow = True` (recommended default):
- Every WRITE automatically gets a background copy to the slow tier
- This means eviction never needs to drain dirty data first (instant eviction)
- Trade-off: higher write amplification, but simpler and faster eviction

## Background Maintenance

Every `bg_interval` seconds (default 5.0s), two functions fire:

1. **bg_evict**: proactively evict if fast tier exceeds a fill threshold (e.g., 90%)
   - Results are *queued* as deferred evictions, consumed on-demand when writes need space
2. **bg_promote**: move hot files from slow to fast tier
   - Results dispatch immediately as background writes

## Warmup Period

The first `warmup_ops` requests are marked `is_warmup = True`:
- They still execute and affect filesystem state (fills the cache)
- They are excluded from metric calculations
- Typical usage: set warmup to ~50% of trace length

## Filesystem State

The evaluator tracks:

| State | Description |
|-------|-------------|
| `_fs[fid]["fast"]` | Size stored in fast tier (None if absent) |
| `_fs[fid]["slow"]` | Size stored in slow tier (None if absent) |
| `_fast_entries[fid]` | `FastEntry(size, dirty)` for each file in fast |
| `_logical_sizes[fid]` | Maximum extent seen: `max(offset + io_size)` across all ops |

The policy receives snapshots of this state (copies, not live references) to make decisions.
