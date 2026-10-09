"""Shared metric types and helpers for cross-implementation comparison."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

WorkMode = Literal["full_matrix", "score_only", "full_vector", "traceback"]


@dataclass
class TimingStats:
    median: float
    samples: list[float] = field(default_factory=list)
    p10: Optional[float] = None
    p90: Optional[float] = None
    stddev: Optional[float] = None
    cv_pct: Optional[float] = None  # coefficient of variation (%)


@dataclass
class ImplResult:
    """One row in the unified comparison table."""

    impl: str
    work_mode: WorkMode
    cells: int
    seconds: float
    gcups: Optional[float]
    speedup_vs_best: Optional[float] = None
    correct: Optional[bool] = None
    score: Optional[float] = None
    score_delta: Optional[float] = None
    matrix_match_pct: Optional[float] = None
    timing: Optional[TimingStats] = None
    note: str = ""
    traceback: str = ""
    boundary_convention: str = ""
    align_query: str = ""
    align_ref: str = ""
    align_comp: str = ""


def percentile(sorted_samples: list[float], p: float) -> float:
    """Linear-interpolation percentile on a pre-sorted sample list."""
    if not sorted_samples:
        return 0.0
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    rank = p * (len(sorted_samples) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_samples[lo]
    frac = rank - lo
    return sorted_samples[lo] * (1.0 - frac) + sorted_samples[hi] * frac


def timing_stats(samples: list[float]) -> TimingStats:
    """Summarize wall-clock samples (seconds)."""
    if not samples:
        return TimingStats(median=0.0, samples=[])
    ordered = sorted(samples)
    med = percentile(ordered, 0.5)
    mean = sum(ordered) / len(ordered)
    var = sum((x - mean) ** 2 for x in ordered) / len(ordered)
    std = math.sqrt(var)
    cv = (std / mean * 100.0) if mean > 0 else None
    return TimingStats(
        median=med,
        samples=list(samples),
        p10=percentile(ordered, 0.10),
        p90=percentile(ordered, 0.90),
        stddev=std,
        cv_pct=cv,
    )


def gcups_from_seconds(cells: int, seconds: float) -> Optional[float]:
    if seconds <= 0 or cells <= 0:
        return None
    return cells / seconds / 1e9


def gcups_from_mb_s(mb_s: float, bytes_per_cell: int) -> Optional[float]:
    """Convert vecrec MB/s (memory bandwidth) to GCUPS.

    vecrec reports ``MB/s = cells * bytes_per_cell / seconds / 1e6``, so
    ``GCUPS = cells / seconds / 1e9 = MB/s / bytes_per_cell``.
    """
    if mb_s <= 0 or bytes_per_cell <= 0:
        return None
    return mb_s / bytes_per_cell


def effective_seconds(r: ImplResult) -> float:
    """Best available median wall time (seconds) for a result row.

    Prefers the timing-stats median, then the recorded ``seconds``, then a
    back-computed time from GCUPS + cell count. Returns ``0.0`` when nothing
    usable is available (e.g. an unbuilt baseline).
    """
    if r.timing is not None and r.timing.median > 0:
        return r.timing.median
    if r.seconds > 0:
        return r.seconds
    if r.gcups is not None and r.gcups > 0 and r.cells > 0:
        return r.cells / r.gcups / 1e9
    return 0.0


def _result_seconds(r: ImplResult) -> float:
    if r.timing is not None and r.timing.median > 0:
        return r.timing.median
    return r.seconds if r.seconds > 0 else 0.0


def apply_speedup_vs_best(results: list[ImplResult]) -> None:
    """Fill ``speedup_vs_best`` in place, comparing within each ``work_mode``.

    Rows with GCUPS compete on throughput (higher is better). Rows without
    GCUPS but with positive wall time compete on speed (lower seconds is better).

    The "best" baseline is drawn only from *correct* rows, so a fast-but-wrong
    implementation never becomes the reference everything else is measured
    against. Incorrect rows are still assigned a speedup relative to that best
    correct row (their ``correct`` column flags the caveat) — this keeps the
    vecrec schedules visible in the comparison even when their kernel carries a
    boundary convention that diverges from the textbook reference.
    """
    by_mode_gcups: dict[str, list[ImplResult]] = {}
    by_mode_time: dict[str, list[ImplResult]] = {}
    for r in results:
        if r.note in ("not built",):
            continue
        if r.gcups is not None and r.gcups > 0:
            by_mode_gcups.setdefault(r.work_mode, []).append(r)
            continue
        sec = _result_seconds(r)
        if sec > 0:
            by_mode_time.setdefault(r.work_mode, []).append(r)

    for group in by_mode_gcups.values():
        correct = [r.gcups or 0.0 for r in group if r.correct is not False]
        best = max(correct) if correct else 0.0
        if best <= 0:
            continue
        for r in group:
            if r.gcups is not None:
                r.speedup_vs_best = r.gcups / best

    for group in by_mode_time.values():
        correct = [_result_seconds(r) for r in group if r.correct is not False]
        best = min(correct) if correct else 0.0
        if best <= 0:
            continue
        for r in group:
            sec = _result_seconds(r)
            if sec > 0:
                r.speedup_vs_best = best / sec
