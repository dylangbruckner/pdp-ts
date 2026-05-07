# Evaluator Overview

The evaluator is a **SimPy discrete-event simulator** that models a two-tier storage system:

- **Fast tier** (e.g., NVMe SSD): limited capacity, high bandwidth, low latency
- **Slow tier** (e.g., HDD): unlimited capacity, lower bandwidth, higher latency

It replays I/O workload traces through a pluggable eviction/placement policy and measures how well that policy utilizes the fast tier.

## What It Simulates

The simulator models realistic storage behavior:

- **Concurrent I/O**: each tier has a finite number of read/write slots (SimPy resources)
- **Bandwidth + latency**: every operation takes `latency + size / bandwidth` seconds
- **Capacity management**: fast tier has a fixed byte budget; writes block or spill when full
- **Background maintenance**: periodic eviction and promotion run on a configurable interval
- **Dirty tracking**: files written only to fast tier must be drained to slow before eviction (unless `always_write_slow=True`)

## Key Outputs

After a simulation run, you get:

| Metric | Description |
|--------|-------------|
| **Hit rate** | Fraction of read bytes served from the fast tier |
| **Response time** | Per-request latency (mean, p50, p99) broken down by tier |
| **Write amplification** | Total bytes written / user-requested write bytes |
| **Per-request log** | CSV with tier served, latency, hit/miss for every operation |

## Architecture

```
Trace File (Thesios parquet or CSV)
        |
        v
  [trace_loader] --> List[Request]
        |
        v
  [Evaluator]
    |-- SimPy environment (event loop)
    |-- StorageTier (fast) with bandwidth/latency/concurrency
    |-- StorageTier (slow) with bandwidth/latency/concurrency
    |-- fast_capacity (SimPy Container tracking free bytes)
    |-- filesystem state (_fs, _fast_entries, _logical_sizes)
    |-- PolicyFunctions (5 decision callbacks)
    |-- MetricsCollector (records every request outcome)
        |
        v
  Output CSV + Summary Metrics
```

## Core Components

| File | Role |
|------|------|
| `src/evaluator.py` | Main simulator: event loop, I/O processes, request dispatch |
| `src/policy.py` | `PolicyFunctions` dataclass + factory functions for built-in policies |
| `src/config.py` | `TierConfig` and `EvaluatorConfig` dataclasses |
| `src/models.py` | Data types: `Request`, `Operation`, `FastEntry`, enums |
| `src/trace_loader.py` | Load Thesios parquet or simple CSV into `List[Request]` |
| `src/metrics.py` | `MetricsCollector` and `GlobalMetrics` |
| `src/data_placement.py` | Policy class implementations (LRU, LFU, S3FIFO, etc.) |
| `src/ml_training.py` | Train LightGBM models for ML-based policies |
| `src/opt_training.py` | Generate OPT (Belady) imitation training data |
