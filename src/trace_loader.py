from __future__ import annotations
import csv
import glob
import os
from typing import List, Optional
from .models import Request, OpType


def load_simple_csv(path: str) -> List[Request]:
    requests: List[Request] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            op = OpType.READ if row["op_type"].upper() == "READ" else OpType.WRITE
            requests.append(Request(
                arrival_time=float(row["timestamp"]),
                op_type=op,
                file_id=row["file_id"],
                offset=int(row.get("offset", 0)),
                size=int(row["size_bytes"]),
                is_warmup=bool(int(row.get("is_warmup", 0))),
                raw=row,
            ))
    return requests


def load_thesios_csv(path: str, max_rows: Optional[int] = None,
                     warmup_rows: int = 0) -> List[Request]:
    """Load Thesios-format CSV trace file(s).

    path: single file or directory. If directory, loads all files sorted by name.
    """
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*")))
    else:
        files = [path]

    requests: List[Request] = []
    count = 0
    for fpath in files:
        if max_rows is not None and count >= max_rows:
            break
        with open(fpath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if max_rows is not None and count >= max_rows:
                    break
                size = int(row["request_io_size_bytes"])
                if size == 0:
                    continue
                op = OpType.READ if row["op_type"].upper() == "READ" else OpType.WRITE
                requests.append(Request(
                    arrival_time=float(row["start_time"]),
                    op_type=op,
                    file_id=row["filename"],
                    offset=int(row["file_offset"]),
                    size=size,
                    is_warmup=(count < warmup_rows),
                    raw=row,
                ))
                count += 1

    requests.sort(key=lambda r: r.arrival_time)
    return requests
