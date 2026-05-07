"""Per-file feature tracking for ML-guided eviction."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .models import OpType

RECENT_CAP = 32
RECENT_WINDOW = 60.0
SHORT_WINDOW = 10.0
DECAY_ALPHA = 0.1  # half-life ~7s

NUM_FEATURES = 16


@dataclass
class FileState:
    size: int
    enter_time: float
    last_access_time: float
    second_last_access_time: float = 0.0
    access_count: int = 0
    read_count: int = 0
    write_count: int = 0
    last_op_was_write: int = 0
    unique_offsets: Set[int] = field(default_factory=set)
    recent_accesses: deque = field(default_factory=lambda: deque(maxlen=RECENT_CAP))
    application_id: int = 0
    service_class_id: int = 0
    file_creation_time: float = 0.0


_app_to_id: Dict[str, int] = {}
_svc_to_id: Dict[str, int] = {"THROUGHPUT_ORIENTED": 0, "LATENCY_SENSITIVE": 1, "OTHER": 2}


def encode_application(app: str) -> int:
    if app not in _app_to_id:
        _app_to_id[app] = len(_app_to_id)
    return _app_to_id[app]


def encode_service_class(svc: str) -> int:
    return _svc_to_id.get(svc, 2)


class FileTracker:
    def __init__(self, recent_window: float = RECENT_WINDOW, decay_alpha: float = DECAY_ALPHA):
        self._state: Dict[str, FileState] = {}
        self._recent_window = recent_window
        self._decay_alpha = decay_alpha

    def add(self, file_id: str, sim_time: float, size: int,
            raw: Optional[dict] = None) -> None:
        if file_id in self._state:
            self._state[file_id].size = size
            self._state[file_id].enter_time = sim_time
            return
        app_id, svc_id, c_time = 0, 0, 0.0
        if raw:
            app_id = encode_application(raw.get("application", ""))
            svc_id = encode_service_class(raw.get("service_class", ""))
            try:
                c_time = float(raw.get("c_time", 0))
            except (ValueError, TypeError):
                c_time = 0.0
        self._state[file_id] = FileState(
            size=size, enter_time=sim_time, last_access_time=sim_time,
            application_id=app_id, service_class_id=svc_id,
            file_creation_time=c_time,
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
        s.last_op_was_write = 1 if op_type == OpType.WRITE else 0
        s.unique_offsets.add(offset)
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
            return np.zeros(NUM_FEATURES)
        return self._build_vector(s, sim_time)

    def all_feature_vectors(self, sim_time: float) -> Tuple[List[str], np.ndarray]:
        fids = list(self._state.keys())
        if not fids:
            return fids, np.empty((0, NUM_FEATURES))
        X = np.empty((len(fids), NUM_FEATURES))
        for i, fid in enumerate(fids):
            X[i] = self._build_vector(self._state[fid], sim_time)
        return fids, X

    def _build_vector(self, s: FileState, sim_time: float) -> np.ndarray:
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

        alpha = self._decay_alpha
        decay_score = sum(
            math.exp(max(-500, -alpha * (sim_time - t))) for t in s.recent_accesses
        )

        file_age = sim_time - s.file_creation_time if s.file_creation_time > 0 else 0.0
        time_since_2nd = sim_time - s.second_last_access_time if s.second_last_access_time > 0 else 0.0

        return np.array([
            sim_time - s.last_access_time,   # 0: time_since_last_access
            s.access_count,                   # 1: access_count
            recent_freq,                      # 2: recent_access_freq (60s)
            s.size,                           # 3: file_size
            rw_ratio,                         # 4: read_write_ratio
            sim_time - s.enter_time,          # 5: time_in_fast_tier
            mean_gap,                         # 6: inter_access_gap_mean
            std_gap,                          # 7: inter_access_gap_std
            decay_score,                      # 8: exponential_decay_score
            s.application_id,                 # 9: application_id
            s.service_class_id,               # 10: service_class_id
            file_age,                         # 11: file_age
            time_since_2nd,                   # 12: time_since_second_last_access
            short_freq,                       # 13: access_count_last_10s
            s.last_op_was_write,              # 14: last_op_was_write
            len(s.unique_offsets),            # 15: unique_offsets_accessed
        ])

    @property
    def tracked_files(self) -> Dict[str, FileState]:
        return self._state

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "time_since_last_access", "access_count", "recent_access_freq",
            "file_size", "read_write_ratio", "time_in_fast_tier",
            "inter_access_gap_mean", "inter_access_gap_std",
            "exponential_decay_score", "application_id", "service_class_id",
            "file_age", "time_since_second_last_access", "access_count_last_10s",
            "last_op_was_write", "unique_offsets_accessed",
        ]
