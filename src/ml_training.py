"""
Training data generation and model training for ML-guided eviction.

Usage:
  python -m src.ml_training --trace traces/workload2/raw/ --output models/coldness.joblib
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

from .ml_features import FileTracker, NUM_FEATURES
from .models import OpType, Request
from .trace_loader import load_thesios_csv


def generate_training_data(
    requests: List[Request],
    fast_capacity: float = 512 * 1024**2,
    max_target: float = 600.0,
    return_timestamps: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (feature_vector, time_to_next_access) pairs from a request trace."""
    sorted_reqs = sorted(requests, key=lambda r: r.arrival_time)

    next_access: List[float] = [float("inf")] * len(sorted_reqs)
    last_seen: Dict[str, float] = {}
    for i in range(len(sorted_reqs) - 1, -1, -1):
        fid = sorted_reqs[i].file_id
        if fid in last_seen:
            next_access[i] = last_seen[fid]
        last_seen[fid] = sorted_reqs[i].arrival_time

    tracker = FileTracker()
    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    t_list: List[float] = []
    tracked_size = 0.0

    for i, req in enumerate(sorted_reqs):
        fid = req.file_id
        in_tracker = fid in tracker.tracked_files

        if req.is_warmup:
            if not in_tracker:
                tracker.add(fid, req.arrival_time, req.size, raw=req.raw)
                tracked_size += req.size
            tracker.update(fid, req.arrival_time, req.op_type, req.size, req.offset)
            continue

        if in_tracker:
            vec = tracker.feature_vector(fid, req.arrival_time)
            time_to_next = min(next_access[i] - req.arrival_time, max_target)
            X_list.append(vec)
            y_list.append(time_to_next)
            t_list.append(req.arrival_time)

        tracker.update(fid, req.arrival_time, req.op_type, req.size)

        if not in_tracker:
            if tracked_size < fast_capacity or req.op_type == OpType.WRITE:
                tracker.add(fid, req.arrival_time, req.size, raw=req.raw)
                tracked_size += req.size

    X = np.array(X_list) if X_list else np.empty((0, NUM_FEATURES))
    y = np.array(y_list) if y_list else np.empty(0)
    if return_timestamps:
        ts = np.array(t_list) if t_list else np.empty(0)
        return X, y, ts
    return X, y


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_path: str,
    params: Optional[dict] = None,
):
    """Train a LightGBM regressor on log(1+time_to_next_access)."""
    import lightgbm as lgb

    y_log = np.log1p(y)

    defaults = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        verbose=-1,
    )
    if params:
        defaults.update(params)

    model = lgb.LGBMRegressor(**defaults)
    model.fit(X, y_log)
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")
    print(f"  samples: {len(y)}, target mean: {y.mean():.2f}s, median: {np.median(y):.2f}s")
    print(f"  features: {FileTracker.feature_names()}")
    return model


def _make_relevance_labels(y: np.ndarray) -> np.ndarray:
    """Bucket time_to_next_access into relevance levels. Higher = colder."""
    labels = np.zeros(len(y), dtype=np.int32)
    labels[y > 1] = 1
    labels[y > 10] = 2
    labels[y > 60] = 3
    labels[y > 300] = 4
    return labels


def _make_groups(X: np.ndarray, timestamps: np.ndarray, window: float = 10.0) -> np.ndarray:
    """Assign samples to groups by time window for LambdaRank."""
    group_ids = (timestamps // window).astype(np.int64)
    _, counts = np.unique(group_ids, return_counts=True)
    return counts


def train_ranking_model(
    X: np.ndarray,
    y: np.ndarray,
    model_path: str,
    timestamps: Optional[np.ndarray] = None,
    params: Optional[dict] = None,
    group_window: float = 10.0,
):
    """Train LGBMRanker with LambdaRank. Higher score = colder file."""
    import lightgbm as lgb

    relevance = _make_relevance_labels(y)

    if timestamps is None:
        n = len(y)
        timestamps = np.arange(n, dtype=np.float64)
        group_window = max(1, n // 200)

    group_sizes = _make_groups(X, timestamps, group_window)
    assert group_sizes.sum() == len(y), "group sizes must sum to n_samples"

    defaults = dict(
        objective="lambdarank",
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        verbose=-1,
        label_gain=[0, 1, 3, 7, 15],
    )
    if params:
        defaults.update(params)

    model = lgb.LGBMRanker(**defaults)
    model.fit(X, relevance, group=group_sizes)
    joblib.dump(model, model_path)
    print(f"Saved ranking model to {model_path}")
    print(f"  samples: {len(y)}, groups: {len(group_sizes)}")
    print(f"  relevance distribution: {np.bincount(relevance)}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train ML coldness model")
    parser.add_argument("--trace", required=True, help="trace file or directory")
    parser.add_argument("--output", default="models/coldness.joblib")
    parser.add_argument("--max-target", type=float, default=600.0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--warmup-rows", type=int, default=5_000)
    parser.add_argument("--fast-capacity", type=int, default=512 * 1024**2)
    parser.add_argument("--ranking", action="store_true", help="use LambdaRank instead of regression")
    parser.add_argument("--group-window", type=float, default=10.0)
    args = parser.parse_args()

    print(f"Loading trace: {args.trace} (max_rows={args.max_rows})")
    reqs = load_thesios_csv(args.trace, max_rows=args.max_rows, warmup_rows=args.warmup_rows)
    print(f"  {len(reqs)} requests loaded")

    print(f"Generating training data (max_target={args.max_target}s)...")
    X, y, ts = generate_training_data(
        reqs, fast_capacity=args.fast_capacity,
        max_target=args.max_target, return_timestamps=True,
    )
    print(f"  {X.shape[0]} samples, mean={y.mean():.2f}s, median={np.median(y):.2f}s")

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.ranking:
        train_ranking_model(X, y, args.output, timestamps=ts, group_window=args.group_window)
    else:
        train_model(X, y, args.output)


if __name__ == "__main__":
    main()
