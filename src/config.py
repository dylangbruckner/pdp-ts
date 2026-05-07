from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TierConfig:
    read_bandwidth: float       # bytes/sec
    write_bandwidth: float      # bytes/sec
    read_latency: float         # seconds
    write_latency: float        # seconds
    max_concurrent_reads: int
    max_concurrent_writes: int
    capacity: float = float("inf")  # bytes; inf = no limit (slow tier)


@dataclass
class EvaluatorConfig:
    fast_tier: TierConfig
    slow_tier: TierConfig
    trace_file: str
    output_file: str
    warmup_ops: int = 0
    # auto-add background slow write for every write (enables instant eviction)
    always_write_slow: bool = True
    # "wait": block on fast_capacity until space is free
    # "spill": redirect write to slow tier immediately if fast is full
    on_fast_full: str = "wait"
    # inject wall-clock policy decision time into simulation timeline
    model_decision_time: bool = False
