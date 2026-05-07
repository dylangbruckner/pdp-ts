from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Any


class OpType(Enum):
    READ = auto()
    WRITE = auto()
    EVICT = auto()       # remove file from fast tier (data assumed safe on slow)
    MIGRATE = auto()     # copy file between tiers


class Tier(Enum):
    FAST = "fast"
    SLOW = "slow"


class Priority(Enum):
    HIGH = 0
    LOW = 1


@dataclass
class Request:
    arrival_time: float     # seconds (absolute timestamp from trace)
    op_type: OpType
    file_id: str
    offset: int
    size: int               # bytes
    is_warmup: bool = False
    raw: Any = field(default=None, repr=False)  # original trace row


@dataclass
class Operation:
    """Instruction returned by DataPlacementAlgorithm to the evaluator."""
    op_type: OpType
    tier: Tier
    file_id: str
    size: int
    offset: int = 0
    priority: Priority = Priority.LOW
    dest_tier: Optional[Tier] = None   # only for MIGRATE (source = tier, dest = dest_tier)
    primary: bool = True               # False = fire-and-forget background op


@dataclass
class RequestRecord:
    """One row in the per-request output CSV."""
    request_type: str
    request_size: int
    served_tier: str
    arrival_time: float
    completion_time: float
    hit: bool               # True = served from fast tier
    is_warmup: bool
    dpa_decision_time_s: float   # wall-clock seconds DPA spent on on_request


@dataclass
class FastEntry:
    size: int
    dirty: bool  # True = no confirmed slow copy (must drain before eviction if aws=False)
