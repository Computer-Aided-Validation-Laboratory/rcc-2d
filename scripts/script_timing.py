"""Append portable per-case timing records for render and analysis scripts."""

from __future__ import annotations

import csv
import os
import socket
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Iterator


def physical_core_count() -> int:
    """Return physical cores on Linux, with a logical-core fallback elsewhere."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        physical_id = "0"
        pairs: set[tuple[str, str]] = set()
        for line in cpuinfo.read_text().splitlines():
            if not line.strip():
                physical_id = "0"
                continue
            key, separator, value = line.partition(":")
            if not separator:
                continue
            if key.strip() == "physical id":
                physical_id = value.strip()
            elif key.strip() == "core id":
                pairs.add((physical_id, value.strip()))
        if pairs:
            return len(pairs)
    return os.cpu_count() or 1


class ScriptTimer:
    """Append one row per named case, including failed case attempts."""

    def __init__(self, script_path: str | Path) -> None:
        stem = Path(script_path).stem
        self.cores = physical_core_count()
        self.path = Path("out") / f"Cores{self.cores}_{stem}_time.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        case: str,
        elapsed_seconds: float,
        status: str = "completed",
        utc_started: str | None = None,
    ) -> None:
        new_file = not self.path.exists()
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "utc_started", "case", "elapsed_seconds", "status",
                    "physical_cores", "logical_cores", "hostname", "script",
                ),
            )
            if new_file:
                writer.writeheader()
            writer.writerow(
                {
                    "utc_started": utc_started or datetime.now(timezone.utc).isoformat(),
                    "case": case,
                    "elapsed_seconds": f"{elapsed_seconds:.6f}",
                    "status": status,
                    "physical_cores": self.cores,
                    "logical_cores": os.cpu_count() or 1,
                    "hostname": socket.gethostname(),
                    "script": Path(sys.argv[0]).name,
                }
            )

    @contextmanager
    def case(self, case: str) -> Iterator[None]:
        start = perf_counter()
        utc_started = datetime.now(timezone.utc).isoformat()
        try:
            yield
        except BaseException:
            self.record(case, perf_counter() - start, "failed", utc_started)
            raise
        else:
            self.record(case, perf_counter() - start, utc_started=utc_started)


def timed_call(timer: ScriptTimer, case: str, function, /, *args, **kwargs):
    """Execute a single render/analysis unit and append its elapsed time."""
    with timer.case(case):
        return function(*args, **kwargs)
