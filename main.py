"""
Entry point for the storage tier evaluator.

Usage:
    python main.py --trace traces/example/example.csv --output results.csv

The fast tier defaults to a 16 TB NVMe-class device.
The slow tier defaults to an HDD-class device with unlimited capacity.
"""
import argparse

from src.config import EvaluatorConfig, TierConfig
from src.data_placement import LRUPolicy
from src.evaluator import Evaluator
from src.trace_loader import load_simple_csv

GB = 1024 ** 3
TB = 1024 ** 4


def build_default_config(trace: str, output: str) -> EvaluatorConfig:
    fast = TierConfig(
        read_bandwidth=3.5 * GB,
        write_bandwidth=3.0 * GB,
        read_latency=0.00002,       # 20 µs
        write_latency=0.00002,
        max_concurrent_reads=32,
        max_concurrent_writes=32,
        capacity=16 * TB,
    )
    slow = TierConfig(
        read_bandwidth=200 * 1024 * 1024,
        write_bandwidth=150 * 1024 * 1024,
        read_latency=0.005,         # 5 ms
        write_latency=0.005,
        max_concurrent_reads=8,
        max_concurrent_writes=8,
    )
    return EvaluatorConfig(fast_tier=fast, slow_tier=slow,
                           trace_file=trace, output_file=output)


def main():
    parser = argparse.ArgumentParser(description="Two-tier storage evaluator")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", default="results.csv")
    parser.add_argument("--policy", choices=["lru"], default="lru")
    args = parser.parse_args()

    cfg = build_default_config(args.trace, args.output)
    requests = load_simple_csv(args.trace)

    dpa = LRUPolicy(fast_capacity=cfg.fast_tier.capacity)
    ev = Evaluator(cfg, dpa, requests)
    metrics = ev.run()

    summary = metrics.global_metrics.summary()
    print("=== Simulation complete ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    wa = metrics.global_metrics.write_amplification(metrics._requested_write_bytes)
    print(f"  write_amplification: {round(wa, 4)}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
