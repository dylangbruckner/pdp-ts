"""Feature tracker using only universally generalizable features."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .models import OpType

RECENT_CAP = 32
RECENT_WINDOW = 60.0
SHORT_WINDOW = 10.0
DECAY_ALPHA = 0.1

NUM_ROBUST_FEATURES = 10


@dataclass
class RobustFileState:
    size: int
    last_access_time: float
    second_last_access_time: float = 0.0
    access_count: int = 0
    read_count: int = 0
    write_count: int = 0
    recent_accesses: deque = field(default_factory=lambda: deque(maxlen=RECENT_CAP))


class RobustTracker:
    def __init__(self, recent_window: float = RECENT_WINDOW, decay_alpha: float = DECAY_ALPHA):
        self._state: Dict[str, RobustFileState] = {}
        self._recent_window = recent_window
        self._decay_alpha = decay_alpha

    def add(self, file_id: str, sim_time: float, size: int, **kwargs) -> None:
        if file_id in self._state:
            self._state[file_id].size = size
            return
        self._state[file_id] = RobustFileState(
            size=size, last_access_time=sim_time,
        )

    def update(self, file_id: str, sim_time: float, op_type: OpType, size: int,
               offset: int = 0) -> None:
        s = self._state.get(file_id)
        if s is None:
            return
        s.second_last_access_time = s.last_access_time
        s.last_access_time = sim_time
        s.access_count += 1
        s.size = size
        if op_type == OpType.READ:
            s.read_count += 1
        else:
            s.write_count += 1
        s.recent_accesses.append(sim_time)

    def remove(self, file_id: str) -> None:
        self._state.pop(file_id, None)

    def feature_vector(self, file_id: str, sim_time: float) -> np.ndarray:
        s = self._state.get(file_id)
        if s is None:
            return np.zeros(NUM_ROBUST_FEATURES)
        return self._build_vector(s, sim_time)

    def all_feature_vectors(self, sim_time: float) -> Tuple[List[str], np.ndarray]:
        fids = list(self._state.keys())
        if not fids:
            return fids, np.empty((0, NUM_ROBUST_FEATURES))
        X = np.empty((len(fids), NUM_ROBUST_FEATURES))
        for i, fid in enumerate(fids):
            X[i] = self._build_vector(self._state[fid], sim_time)
        return fids, X

    def _build_vector(self, s: RobustFileState, sim_time: float) -> np.ndarray:
        cutoff_long = sim_time - self._recent_window
        cutoff_short = sim_time - SHORT_WINDOW
        recent_freq = sum(1 for t in s.recent_accesses if t > cutoff_long)
        short_freq = sum(1 for t in s.recent_accesses if t > cutoff_short)
        total = s.read_count + s.write_count
        rw_ratio = s.read_count / max(1, total)

        times = list(s.recent_accesses)
        gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
        mean_gap = np.mean(gaps) if gaps else 0.0
        std_gap = np.std(gaps) if len(gaps) > 1 else 0.0

        decay_score = sum(
            math.exp(max(-500, -self._decay_alpha * (sim_time - t)))
            for t in s.recent_accesses
        )

        time_since_2nd = sim_time - s.second_last_access_time if s.second_last_access_time > 0 else 0.0

        return np.array([
            sim_time - s.last_access_time,   # 0: time_since_last_access
            s.access_count,                   # 1: access_count
            recent_freq,                      # 2: recent_access_freq (60s)
            s.size,                           # 3: file_size
            rw_ratio,                         # 4: read_write_ratio
            mean_gap,                         # 5: inter_access_gap_mean
            std_gap,                          # 6: inter_access_gap_std
            decay_score,                      # 7: exponential_decay_score
            short_freq,                       # 8: access_count_last_10s
            time_since_2nd,                   # 9: time_since_second_last_access
        ])

    @property
    def tracked_files(self) -> Dict[str, RobustFileState]:
        return self._state

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "time_since_last_access", "access_count", "recent_access_freq",
            "file_size", "read_write_ratio", "inter_access_gap_mean",
            "inter_access_gap_std", "exponential_decay_score",
            "access_count_last_10s", "time_since_second_last_access",
        ]
