# Setup and Usage

## Installation

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| simpy | Discrete-event simulation engine |
| pandas | Trace loading and data manipulation |
| pyarrow | Reading Thesios parquet trace files |
| lightgbm | ML model training (gradient boosted trees) |
| scikit-learn | ML utilities and metrics |
| joblib | Model serialization |
| numpy | Numerical operations |

## Running with the CLI

```bash
python main.py --trace traces/workload2/cluster_1_group_1 --policy lru --output results.csv
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--trace` | (required) | Path to trace file or directory of parquet shards |
| `--policy` | `lru` | Policy name (see available policies below) |
| `--output` | `results.csv` | Output CSV path |

## Running Programmatically

```python
from src.config import TierConfig, EvaluatorConfig
from src.trace_loader import load_thesios_csv
from src.policy import make_lru_policy
from src.evaluator import Evaluator

# 1. Configure tiers
fast = TierConfig(
    read_bandwidth=3.5e9,       # 3.5 GB/s
    write_bandwidth=3.0e9,      # 3.0 GB/s
    read_latency=0.00002,       # 20 us
    write_latency=0.00002,
    max_concurrent_reads=32,
    max_concurrent_writes=32,
    capacity=10 * 1024**3,      # 10 GB fast tier
)

slow = TierConfig(
    read_bandwidth=200e6,       # 200 MB/s
    write_bandwidth=150e6,      # 150 MB/s
    read_latency=0.005,         # 5 ms
    write_latency=0.005,
    max_concurrent_reads=8,
    max_concurrent_writes=8,
    # capacity defaults to inf (unlimited)
)

config = EvaluatorConfig(
    fast_tier=fast,
    slow_tier=slow,
    trace_file="traces/workload2/cluster_1_group_1",
    output_file="my_results.csv",
    warmup_ops=70000,           # first 70k ops are warmup
    always_write_slow=True,     # recommended: enables instant eviction
    on_fast_full="wait",        # block writes until eviction frees space
)

# 2. Load trace
requests = load_thesios_csv(
    config.trace_file,
    max_rows=142000,            # one shard ~142k rows
    warmup_rows=config.warmup_ops,
)

# 3. Create policy
policy = make_lru_policy(fast_capacity=fast.capacity)

# 4. Run simulation
evaluator = Evaluator(config, policy, requests)
metrics = evaluator.run()

# 5. Read results
summary = metrics.global_metrics.summary()
print(f"Hit rate: {summary['hit_rate']:.4f}")
print(f"Mean response time: {summary['mean_response_time_s']:.6f}s")
```

## Using Thesios Traces

The traces live in `traces/workload2/` with three clusters:

```
traces/workload2/
  cluster_1_group_1/   # 50 parquet shards (~142k rows each)
  cluster_1_group_2/   # alternative time period
  cluster_3_group_1/   # different cluster
```

Each parquet file contains columns: `fid`, `op_type`, `file_offset`, `request_io_size_bytes`, `start_time`, plus metadata columns.

### Loading options

```python
# Load a single shard
requests = load_thesios_csv("traces/workload2/cluster_1_group_1/cluster1_16TB_20240115_data-00000-of-00100")

# Load entire cluster directory (all shards concatenated, sorted by time)
requests = load_thesios_csv("traces/workload2/cluster_1_group_1")

# Load with row limit and warmup
requests = load_thesios_csv(
    "traces/workload2/cluster_1_group_1",
    max_rows=142000,
    warmup_rows=71000,
)
```

### Recommended evaluation methodology

- **Train** on the first half of shards (e.g., shards 0-3)
- **Evaluate** on a later shard (e.g., shard 4+) to test generalization
- **Warmup** = first 50% of the evaluation shard (fills cache to steady state)

## Available Built-in Policies

| Policy | Factory Function | Description |
|--------|-----------------|-------------|
| LRU | `make_lru_policy` | Least Recently Used eviction |
| Size-Aware LRU | `make_size_aware_lru_policy` | LRU weighted by file size |
| LFU | `make_lfu_policy` | Least Frequently Used eviction |
| S3FIFO | `make_s3fifo_policy` | Small/Main/Ghost three-queue FIFO |
| ML Regression | `make_ml_policy` | LightGBM coldness predictor |
| Hybrid LRU+ML | `make_hybrid_policy` | LRU with ML protection filter |
| Online ML | `make_online_ml_policy` | ML with periodic retraining |
| OPT (Oracle) | `make_opt_policy` | Belady's optimal (requires full trace) |
| Fusion | `make_fusion_policy` | Score fusion of multiple signals |
| Placement ML | `make_placement_ml_policy` | ML-driven initial placement |
| Adaptive ML | `make_adaptive_ml_policy` | ML with adaptive thresholds |

### ML policies require a trained model

```python
from src.policy import make_ml_policy

policy = make_ml_policy(
    fast_capacity=10 * 1024**3,
    model_path="path/to/model.joblib",
)
```

See `src/ml_training.py` and `src/opt_training.py` for model training.

## Configuration Knobs

| Parameter | Effect |
|-----------|--------|
| `always_write_slow=True` | Every write gets a slow-tier copy. Enables instant eviction (no drain needed). Increases write amp ~2x. |
| `always_write_slow=False` | Writes only go to the tier the policy selects. Eviction of dirty files requires a drain write first. |
| `on_fast_full="wait"` | When fast tier is full, block the write and call `evict_bytes` to make space. |
| `on_fast_full="spill"` | When fast tier is full, redirect the write to the slow tier immediately. |
| `warmup_ops=N` | First N requests fill the cache but don't count toward metrics. |
| `bg_interval` | How often (sim seconds) background eviction/promotion runs. Default 5.0. |
