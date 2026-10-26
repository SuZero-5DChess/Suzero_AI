"""How often each self-play sample has been trained on.

The replay backends keep one counter per *live* row: a row that leaves the
replay window through trimming loses its counter, so the summary always
describes the samples the trainer can currently reach.  Counting is a
diagnostic: a missing or out-of-range row must never stop training.

The counters also decide what trimming may evict.  A row is discarded only
after the trainer has served it at least once; rows that were never reached
stay in the window even when it grows past its target.  Because eviction is
front-only, an untrained row also protects every row behind it.

Rows are identified per backend:

* :class:`~suzero_5dchess.game_store.GameStore` serves rows by window
  position, while its persisted counters use ``(game_id, local row)`` so a
  front trim keeps the surviving rows aligned;
* :class:`~suzero_5dchess.replay.ReplayBuffer` uses the position in its FIFO
  blob list, saved together with the blobs.

Data sources record a count only when a batch is actually served, so samples
that never reach the trainer stay at zero.
"""

from __future__ import annotations

from math import ceil
from typing import Iterable, Sequence

import numpy as np

#: On-disk counter dtype.  ``uint32`` is enough for any realistic number of
#: passes over one sample and halves the sidecar size.
COUNT_DTYPE = np.uint32

#: Counts at or above this value are folded into one ``"10+"`` bucket, which
#: keeps the reported histogram readable for long runs.
HISTOGRAM_LIMIT = 10

HISTOGRAM_KEYS = [str(value) for value in range(HISTOGRAM_LIMIT)] + [
    f"{HISTOGRAM_LIMIT}+"
]


def new_counts(size: int = 0) -> list[int]:
    """Return a zeroed counter list for ``size`` rows."""

    if size < 0:
        raise ValueError("counter size must be non-negative")
    return [0] * int(size)


def summarize(counts: Sequence[int]) -> dict[str, object]:
    """Describe how many times the live samples were trained.

    ``median`` is the lower median and ``p90`` uses nearest-rank, so both are
    exact per-sample counts rather than interpolated values.
    """

    total = len(counts)
    if total == 0:
        return {
            "samples": 0,
            "trained": 0,
            "untrained": 0,
            "total_trains": 0,
            "mean": 0.0,
            "median": 0,
            "p90": 0,
            "max": 0,
            "histogram": {key: 0 for key in HISTOGRAM_KEYS},
        }

    values = np.fromiter((int(value) for value in counts), dtype=np.int64, count=total)
    if values.size and int(values.min()) < 0:
        raise ValueError("training counts must be non-negative")

    total_trains = int(values.sum())
    trained = int(np.count_nonzero(values))
    bounded = np.minimum(values, HISTOGRAM_LIMIT)
    frequency = np.bincount(bounded, minlength=HISTOGRAM_LIMIT + 1)
    histogram = {key: int(frequency[index]) for index, key in enumerate(HISTOGRAM_KEYS)}

    ordered = np.sort(values)
    median = int(ordered[(total - 1) // 2])
    p90 = int(ordered[min(total - 1, max(0, ceil(0.9 * total) - 1))])
    return {
        "samples": total,
        "trained": trained,
        "untrained": total - trained,
        "total_trains": total_trains,
        "mean": round(total_trains / total, 4),
        "median": median,
        "p90": p90,
        "max": int(ordered[-1]),
        "histogram": histogram,
    }


def record(counts: np.ndarray | list[int], rows: Iterable[int]) -> int:
    """Increment ``counts`` for each row; ignore rows outside the window.

    Returns the number of rows that were actually counted.  Silent skipping
    keeps a stale shuffle round from breaking a training run after unusual
    replay mutations; the caller can compare the return value if needed.
    """

    recorded = 0
    size = len(counts)
    for value in rows:
        index = int(value)
        if 0 <= index < size:
            counts[index] += 1
            recorded += 1
    return recorded


def evictable_prefix(counts: Sequence[int] | np.ndarray, limit: int) -> int:
    """Leading rows that may be evicted, capped at ``limit``.

    Eviction only removes rows from the front of the window, and only rows the
    trainer has already served at least once.  An untrained front row therefore
    stops the whole trim: the window keeps that row and everything behind it
    until the trainer reaches them.
    """

    requested = max(0, int(limit))
    if not requested:
        return 0
    values = np.asarray(counts[:requested], dtype=np.int64)
    if values.size == 0:
        return 0
    untrained = np.flatnonzero(values == 0)
    return int(untrained[0]) if untrained.size else int(values.size)


def untrained_first(
    order: np.ndarray | Sequence[int], counts: Sequence[int] | np.ndarray
) -> np.ndarray:
    """Stable partition: never-trained rows before rows with a training pass.

    Called after a round has been shuffled.  Each side keeps its shuffled
    order, so a fresh window is untouched, while a backlog of untrained rows
    is served before the budget is spent on rows that already had a pass.
    """

    rows = np.asarray(order, dtype=np.int64)
    if rows.size == 0:
        return rows
    values = np.asarray(counts, dtype=np.int64)
    if values.size == 0:
        return rows
    trained = values[rows] > 0
    if trained.all() or not trained.any():
        return rows
    return np.concatenate((rows[~trained], rows[trained]))


__all__ = [
    "COUNT_DTYPE",
    "HISTOGRAM_LIMIT",
    "HISTOGRAM_KEYS",
    "evictable_prefix",
    "new_counts",
    "record",
    "summarize",
    "untrained_first",
]
