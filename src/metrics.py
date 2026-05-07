from __future__ import annotations
import csv
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from .models import RequestRecord, Tier


@dataclass
class GlobalMetrics:
    total_requests: int = 0
    hits: int = 0
    bytes_written_fast: int = 0
    bytes_written_slow: int = 0
    bytes_read_fast: int = 0
    bytes_read_slow: int = 0
    total_fast_stall_time: float = 0.0
    total_dpa_decision_time: float = 0.0
    dpa_background_activations: int = 0
    dpa_background_times: List[float] = field(default_factory=list)
    _latencies_all: List[float] = field(default_factory=list)
    _latencies_hit: List[float] = field(default_factory=list)
    _latencies_miss: List[float] = field(default_factory=list)
    _latencies_fast: List[float] = field(default_factory=list)
    _latencies_slow: List[float] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        reads = self.bytes_read_fast + self.bytes_read_slow
        return self.bytes_read_fast / reads if reads else 0.0

    @property
    def write_amplification_fast(self) -> float:
        return self.bytes_written_fast / max(1, self.bytes_written_fast)

    def write_amplification(self, requested_write_bytes: int) -> float:
        total_written = self.bytes_written_fast + self.bytes_written_slow
        return total_written / max(1, requested_write_bytes)

    def _percentiles(self, data: List[float]) -> dict:
        if not data:
            return {"mean": 0.0, "p50": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
        arr = np.array(data)
        return {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    def latency_summary(self) -> dict:
        return {
            "all": self._percentiles(self._latencies_all),
            "hit": self._percentiles(self._latencies_hit),
            "miss": self._percentiles(self._latencies_miss),
            "fast_tier": self._percentiles(self._latencies_fast),
            "slow_tier": self._percentiles(self._latencies_slow),
        }

    def summary(self) -> dict:
        lat = self.latency_summary()
        return {
            "total_requests": self.total_requests,
            "hit_rate": round(self.hit_rate, 4),
            "bytes_written_fast": self.bytes_written_fast,
            "bytes_written_slow": self.bytes_written_slow,
            "bytes_read_fast": self.bytes_read_fast,
            "bytes_read_slow": self.bytes_read_slow,
            "total_fast_stall_time_s": round(self.total_fast_stall_time, 6),
            "total_dpa_decision_time_s": round(self.total_dpa_decision_time, 6),
            "dpa_background_activations": self.dpa_background_activations,
            "mean_response_time_s": round(lat["all"]["mean"], 6),
            "p50_response_time_s": round(lat["all"]["p50"], 6),
            "p99_response_time_s": round(lat["all"]["p99"], 6),
            "mean_response_time_hit_s": round(lat["hit"]["mean"], 6),
            "mean_response_time_miss_s": round(lat["miss"]["mean"], 6),
            "p99_response_time_hit_s": round(lat["hit"]["p99"], 6),
            "p99_response_time_miss_s": round(lat["miss"]["p99"], 6),
            "mean_latency_fast_s": round(lat["fast_tier"]["mean"], 6),
            "mean_latency_slow_s": round(lat["slow_tier"]["mean"], 6),
            "p99_latency_fast_s": round(lat["fast_tier"]["p99"], 6),
            "p99_latency_slow_s": round(lat["slow_tier"]["p99"], 6),
        }


class MetricsCollector:
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.global_metrics = GlobalMetrics()
        self._records: List[RequestRecord] = []
        self._requested_write_bytes: int = 0

    def record_request(self, rec: RequestRecord, fast_stall: float = 0.0) -> None:
        self._records.append(rec)
        gm = self.global_metrics
        gm.total_requests += 1
        if rec.hit:
            gm.hits += 1
        gm.total_dpa_decision_time += rec.dpa_decision_time_s
        gm.total_fast_stall_time += fast_stall

        if not rec.is_warmup:
            latency = rec.completion_time - rec.arrival_time
            gm._latencies_all.append(latency)
            if rec.hit:
                gm._latencies_hit.append(latency)
            else:
                gm._latencies_miss.append(latency)
            if rec.served_tier == "fast":
                gm._latencies_fast.append(latency)
            else:
                gm._latencies_slow.append(latency)

    def record_write(self, tier: Tier, size: int) -> None:
        if tier == Tier.FAST:
            self.global_metrics.bytes_written_fast += size
        else:
            self.global_metrics.bytes_written_slow += size

    def record_read(self, tier: Tier, size: int) -> None:
        if tier == Tier.FAST:
            self.global_metrics.bytes_read_fast += size
        else:
            self.global_metrics.bytes_read_slow += size

    def record_requested_write(self, size: int) -> None:
        self._requested_write_bytes += size

    def record_background_activation(self, sim_time: float) -> None:
        gm = self.global_metrics
        gm.dpa_background_activations += 1
        gm.dpa_background_times.append(sim_time)

    def flush(self) -> None:
        with open(self.output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "request_type", "request_size", "served_tier",
                "arrival_time", "completion_time", "hit", "is_warmup",
                "dpa_decision_time_s",
            ])
            writer.writeheader()
            for r in self._records:
                writer.writerow({
                    "request_type": r.request_type,
                    "request_size": r.request_size,
                    "served_tier": r.served_tier,
                    "arrival_time": r.arrival_time,
                    "completion_time": r.completion_time,
                    "hit": r.hit,
                    "is_warmup": r.is_warmup,
                    "dpa_decision_time_s": r.dpa_decision_time_s,
                })

        with open(self.output_file, "a") as f:
            f.write("\n# --- summary ---\n")
            for k, v in self.global_metrics.summary().items():
                f.write(f"# {k}: {v}\n")
            wa = self.global_metrics.write_amplification(self._requested_write_bytes)
            f.write(f"# write_amplification: {round(wa, 4)}\n")
            if self.global_metrics.dpa_background_times:
                times_str = ",".join(str(round(t, 3))
                                     for t in self.global_metrics.dpa_background_times[:20])
                f.write(f"# dpa_background_activation_times (first 20): {times_str}\n")
