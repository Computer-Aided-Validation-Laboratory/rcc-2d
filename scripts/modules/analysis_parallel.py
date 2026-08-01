"""Consistent bounded process-pool harness for analysis work items."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections.abc import Callable, Iterable
from multiprocessing.context import BaseContext
from typing import TypeVar

from exp0params_common import CORES


T = TypeVar("T")
R = TypeVar("R")


def analysis_workers(item_count: int) -> int:
    """Return the bounded analysis worker count for this one script."""
    requested = int(os.environ.get("ANALYSIS_WORKERS", str(CORES)))
    return max(1, min(requested, max(1, item_count)))


def run_analysis_jobs(
    label: str,
    jobs: Iterable[T],
    worker: Callable[[T], R],
    *,
    mp_context: BaseContext | None = None,
) -> list[R]:
    """Run independent analysis items in bounded child processes.

    Result collection happens in the parent, keeping CSV writes and aggregate
    state serial while child processes release plotting/NumPy memory on exit.
    """
    items = list(jobs)
    if not items:
        return []
    workers = analysis_workers(len(items))
    print(f"{label}: {len(items)} work items; {workers} analysis workers.")
    if workers == 1:
        return [worker(item) for item in items]
    results: list[R] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as pool:
        futures = [pool.submit(worker, item) for item in items]
        for future in as_completed(futures):
            results.append(future.result())
    return results
