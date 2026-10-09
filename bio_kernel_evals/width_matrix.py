"""Per-recurrence width/mode comparison matrix.

This runner produces a *consistent*, structured view on top of the same native
drivers the main suite runs (see ``evals_common/vecrec_cpp.py`` and the C++
library drivers under ``baselines/``). For every recurrence it emits, per SIMD
element width (I8 / I16 / I32) and per work mode (full matrix + traceback vs
score-only), one table that lists:

  * the RAW median time of every wired library baseline at that width, and
  * the RAW median time of the single fastest *correct* VecRec schedule at that
    width (the best-of over all schedules and ``ele_per_iter`` multiples), and
  * ``vr×`` -- VecRec's speedup vs the fastest library in that width group.

Grouping rule (matches the agreed sample output):
  * I8  tables compare against I8-only library kernels (parasail ``*_8``).
  * I32 tables compare against I32-only library kernels (parasail ``*_32``).
  * I16 tables compare against I16 library kernels (parasail ``*_16``) PLUS the
    non-width-parameterized SIMD libraries (SeqAn3, ksw2, edlib), whose internal
    element width is annotated in the column header.

The timed span for every entry is allocation + DP + traceback (matrix mode) or
allocation + DP (score-only mode) -- the drivers self-time under the same
``baselines::Timer``, so entries within a table are directly comparable (see
``evals_common/vecrec_cpp.py`` module docstring).

Run it::

    uv run python -m bio_kernel_evals.width_matrix              # all recurrences, default sizes
    uv run python -m bio_kernel_evals.width_matrix --sizes 128,256,512
    uv run python -m bio_kernel_evals.width_matrix --recurrence needleman

The process exit code is non-zero if any *correctness-checked library* baseline
fails to validate against the textbook reference (a harness/reference sanity
failure) or if any structural self-check fails. VecRec correctness never gates
the exit code -- an I8 ``FAIL`` from saturation at large N is expected and is
reported inline with a ``✗`` marker, not treated as an error.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

from evals_common.metrics import ImplResult, timing_stats

# ---------------------------------------------------------------------------
# Recurrence registry
# ---------------------------------------------------------------------------


@dataclass
class Recurrence:
    """A single recurrence family to benchmark across a size sweep.

    ``ctor(size)`` builds the :class:`~evals_common.harness.EvalCase` at a square
    problem of that size (with the recurrence's fixed band width baked in).
    """

    label: str
    ctor: Callable[[int], object]
    default_sizes: tuple[int, ...]


def _registry() -> list[Recurrence]:
    from bio_kernel_evals.cases._alignment import SWEEP_SIZES
    from bio_kernel_evals.cases.affine_gap import AffineGap
    from bio_kernel_evals.cases.affine_gap_local import AffineGapLocal
    from bio_kernel_evals.cases.banded_affine import (
        BandedAffine,
        BAND_WIDTH as BW_AFFINE,
    )
    from bio_kernel_evals.cases.banded_needleman_wunsch import (
        BandedNeedlemanWunsch,
        BAND_WIDTH as BW_NW,
    )
    from bio_kernel_evals.cases.banded_smith_waterman import (
        BandedSmithWaterman,
        BAND_WIDTH as BW_SW,
    )
    from bio_kernel_evals.cases.edit_distance import EditDistance
    from bio_kernel_evals.cases.extension_affine import (
        ExtensionAffine,
        BAND_WIDTH as BW_EXTZ,
    )
    from bio_kernel_evals.cases.extension_dual_affine import (
        ExtensionDualAffine,
        BAND_WIDTH as BW_EXTD,
    )
    from bio_kernel_evals.cases.needleman_wunsch import NeedlemanWunsch
    from bio_kernel_evals.cases.smith_waterman import SmithWaterman

    banded_sizes = (128, 256, 512, 1024)
    return [
        Recurrence("needleman-wunsch", lambda n: NeedlemanWunsch(rows=n, cols=n), SWEEP_SIZES),
        Recurrence("smith-waterman", lambda n: SmithWaterman(rows=n, cols=n), SWEEP_SIZES),
        Recurrence("affine-gap", lambda n: AffineGap(rows=n, cols=n), SWEEP_SIZES),
        Recurrence("edit-distance", lambda n: EditDistance(rows=n, cols=n), SWEEP_SIZES),
        Recurrence("affine-gap-local", lambda n: AffineGapLocal(rows=n, cols=n), SWEEP_SIZES),
        Recurrence(
            "banded-needleman-wunsch",
            lambda n: BandedNeedlemanWunsch(rows=n, cols=n, band_width=BW_NW),
            banded_sizes,
        ),
        Recurrence(
            "banded-smith-waterman",
            lambda n: BandedSmithWaterman(rows=n, cols=n, band_width=BW_SW),
            banded_sizes,
        ),
        Recurrence(
            "banded-affine",
            lambda n: BandedAffine(rows=n, cols=n, band_width=BW_AFFINE),
            banded_sizes,
        ),
        Recurrence(
            "extension-affine (full)",
            lambda n: ExtensionAffine(rows=n, cols=n, band_width=0),
            SWEEP_SIZES,
        ),
        Recurrence(
            "extension-affine (bw=%d)" % BW_EXTZ,
            lambda n: ExtensionAffine(rows=n, cols=n, band_width=BW_EXTZ),
            SWEEP_SIZES,
        ),
        Recurrence(
            "extension-dual-affine (full)",
            lambda n: ExtensionDualAffine(rows=n, cols=n, band_width=0),
            SWEEP_SIZES,
        ),
        Recurrence(
            "extension-dual-affine (bw=%d)" % BW_EXTD,
            lambda n: ExtensionDualAffine(rows=n, cols=n, band_width=BW_EXTD),
            SWEEP_SIZES,
        ),
    ]


# ---------------------------------------------------------------------------
# Classification: impl name -> (kind, mode, width)
# ---------------------------------------------------------------------------

WIDTHS: tuple[str, ...] = ("i8", "i16", "i32")
MODES: tuple[tuple[str, str], ...] = (
    ("matrix", "MATRIX + TRACEBACK"),
    ("score", "SCORE-ONLY (O(row) buffer, no traceback)"),
)

# Internal element width annotation for the non-width-parameterized libraries
# that get folded into the I16 column (see module docstring).
_LIB_WIDTH_NOTE: dict[str, str] = {
    "ksw2_gg2_sse": "i8/i16",
    "ksw2_extz2_sse": "i8/i16",
    "ksw2_extz2_sse_approx": "i8/i16",
    "ksw2_extd2_sse": "i8/i16",
    "ksw2_extd2_sse_approx": "i8/i16",
    "ksw2_extf2_sse": "i8",
    "ksw2_ll_i16": "i16",
    "seqan3_nw": "i32",
    "seqan3_sw": "i32",
    "seqan3_nw_banded": "i32",
    "edlib": "bitvec",
    "wfa2_edit": "exact",
    "wfa2_nw": "exact",
    "wfa2_affine": "exact",
}


@dataclass(frozen=True)
class Bucket:
    kind: str  # "vecrec" | "lib"
    mode: str  # "matrix" | "score"
    width: str  # "i8" | "i16" | "i32"


def classify(impl: str) -> Optional[Bucket]:
    """Map an impl name to its (kind, mode, width) bucket, or ``None`` to exclude.

    Excluded: parasail scalar ``stats`` / ``nw_banded`` correctness anchors and
    the ``_64`` / ``_sat`` parasail widths (outside the I8/I16/I32 grouping);
    any Python/library entry we don't recognize.
    """
    if impl.startswith("vecrec"):
        mode = "score" if "_so" in impl else "matrix"
        m = re.search(r"_i(8|16|32)_", impl)
        width = f"i{m.group(1)}" if m else "i16"  # single-width drivers are I16
        return Bucket("vecrec", mode, width)

    if impl.startswith("parasail"):
        if "banded" in impl or impl.endswith("stats"):
            return None
        if impl.endswith("_8"):
            width = "i8"
        elif impl.endswith("_16"):
            width = "i16"
        elif impl.endswith("_32"):
            width = "i32"
        else:  # _64, _sat -> outside the I8/I16/I32 grouping
            return None
        return Bucket("lib", "matrix", width)

    if impl.startswith("seqan3"):
        return Bucket("lib", "matrix", "i16")

    if impl.startswith("ksw2"):
        mode = "score" if ("ll_i16" in impl or "extf2" in impl) else "matrix"
        return Bucket("lib", mode, "i16")

    if impl.startswith("edlib"):
        return Bucket("lib", "matrix", "i16")

    if impl.startswith("wfa2"):
        return Bucket("lib", "matrix", "i16")

    return None


def _lib_sort_key(impl: str) -> tuple[float, str]:
    """Stable column ordering: parasail families, then ksw2, seqan3, edlib, wfa2."""
    if impl.startswith("parasail"):
        if "striped" in impl:
            return (0.0, impl)
        if "scan" in impl:
            return (1.0, impl)
        if "diag" in impl:
            return (2.0, impl)
        return (3.0, impl)
    if impl.startswith("ksw2"):
        base = {
            "gg2": 10.0,
            "extz2": 11.0,
            "extd2": 12.0,
            "extf2": 13.0,
            "ll_i16": 14.0,
        }
        order = 19.0
        for key, val in base.items():
            if key in impl:
                order = val
                break
        if "approx" in impl:
            order += 0.5
        return (order, impl)
    if impl.startswith("seqan3"):
        return (20.0, impl)
    if impl.startswith("edlib"):
        return (30.0, impl)
    if impl.startswith("wfa2"):
        return (40.0, impl)
    return (99.0, impl)


# ---------------------------------------------------------------------------
# Running a case (native drivers only; skips the slow numpy baseline)
# ---------------------------------------------------------------------------


def _median(r: ImplResult) -> float:
    if r.timing is not None and r.timing.median > 0:
        return r.timing.median
    return r.seconds if r.seconds > 0 else 0.0


def gather_case(case) -> tuple[list[ImplResult], Optional[float]]:
    """Run every C++ / VecRec driver wired on ``case`` and return their results.

    Deliberately skips the Python ``baselines()`` (numpy reference loop) because
    it is O(N^2) interpreted and dominates wall time at large N; correctness is
    still checked against ``case.reference()`` directly.
    """
    inputs = case.make_inputs()
    expected_matrix = case.reference(**inputs)
    expected_score = (
        case.optimal_score(expected_matrix) if expected_matrix is not None else None
    )

    results: list[ImplResult] = []
    for cpp in case.cpp_baselines():
        if not cpp.available():
            results.append(
                ImplResult(
                    impl=cpp.name,
                    work_mode="traceback",
                    cells=case.n_elements,
                    seconds=0.0,
                    gcups=None,
                    correct=None,
                    note="not built",
                )
            )
            continue
        try:
            res = cpp.run(inputs)
        except Exception as exc:  # noqa: BLE001 - report and continue
            results.append(
                ImplResult(
                    impl=cpp.name,
                    work_mode="traceback",
                    cells=case.n_elements,
                    seconds=0.0,
                    gcups=None,
                    correct=False,
                    note=f"error: {str(exc).splitlines()[0] if str(exc) else type(exc).__name__}",
                )
            )
            continue

        stats = timing_stats(res.samples) if res.samples else timing_stats([res.seconds])
        correct: Optional[bool] = None
        if expected_score is not None and getattr(cpp, "check_correctness", True):
            correct = abs(float(res.score) - float(expected_score)) <= case.score_atol
        results.append(
            ImplResult(
                impl=cpp.name,
                work_mode="traceback",
                cells=res.cells,
                seconds=res.seconds,
                gcups=None,
                correct=correct,
                score=float(res.score),
                timing=stats,
            )
        )
    return results, expected_score


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class SizePoint:
    size: int
    results: list[ImplResult]
    expected_score: Optional[float]


@dataclass
class RecurrenceData:
    label: str
    points: list[SizePoint] = field(default_factory=list)


def run_recurrence(rec: Recurrence, sizes: tuple[int, ...]) -> RecurrenceData:
    data = RecurrenceData(label=rec.label)
    for n in sizes:
        case = rec.ctor(n)
        results, expected = gather_case(case)
        data.points.append(SizePoint(size=n, results=results, expected_score=expected))
    return data


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt_time(sec: Optional[float]) -> str:
    if sec is None or sec <= 0:
        return "-"
    if sec < 1e-3:
        return f"{sec * 1e6:.1f}\u00b5s"
    if sec < 1.0:
        return f"{sec * 1e3:.3f}ms"
    return f"{sec:.4f}s"


def _short_sched(impl: str) -> str:
    return impl.split(":", 1)[1] if ":" in impl else impl


def _bucket_entries(point: SizePoint, mode: str, width: str) -> tuple[
    list[ImplResult], list[ImplResult]
]:
    """Split a size point's results into (vecrec, library) entries for a bucket."""
    vecrec: list[ImplResult] = []
    libs: list[ImplResult] = []
    for r in point.results:
        b = classify(r.impl)
        if b is None or b.mode != mode or b.width != width:
            continue
        (vecrec if b.kind == "vecrec" else libs).append(r)
    return vecrec, libs


def _best_vecrec(entries: list[ImplResult]) -> tuple[Optional[ImplResult], bool]:
    """Return (best correct-or-fallback entry, flagged) for a size's vecrec set.

    ``flagged`` is True when the fastest schedule failed correctness (so the
    displayed time is the fastest *passing* one, or -- if none passed -- the
    fastest overall marked untrustworthy).
    """
    timed = [r for r in entries if _median(r) > 0]
    if not timed:
        return None, False
    fastest = min(timed, key=_median)
    correct = [r for r in timed if r.correct is not False]
    if not correct:
        return fastest, True  # nothing passed -> show fastest, flag it
    best = min(correct, key=_median)
    flagged = _median(fastest) < _median(best) - 1e-12
    return best, flagged


def _best_lib_time(libs: list[ImplResult]) -> Optional[float]:
    times = [_median(r) for r in libs if r.correct is not False and _median(r) > 0]
    if not times:
        times = [_median(r) for r in libs if _median(r) > 0]
    return min(times) if times else None


def _render_table(title: str, header: list[str], rows: list[list[str]], footers: list[str]) -> list[str]:
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    out = [f"--- {title} ---"]
    out.append("  " + "  ".join(h.rjust(widths[i]) for i, h in enumerate(header)))
    for row in rows:
        out.append("  " + "  ".join(cell.rjust(widths[i]) for i, cell in enumerate(row)))
    out.extend(footers)
    return out


def render_recurrence(data: RecurrenceData) -> list[str]:
    lines: list[str] = ["", "=" * 78, f"RECURRENCE: {data.label}", "=" * 78]
    sizes = [p.size for p in data.points]

    for mode, mode_title in MODES:
        mode_has_any = False
        mode_lines: list[str] = []
        for width in WIDTHS:
            # Column set = union of library impls seen in this bucket across sizes.
            lib_names: set[str] = set()
            has_vecrec = False
            for p in data.points:
                vs, ls = _bucket_entries(p, mode, width)
                has_vecrec = has_vecrec or bool(vs)
                lib_names.update(r.impl for r in ls)
            if not has_vecrec and not lib_names:
                continue
            mode_has_any = True

            ordered_libs = sorted(lib_names, key=_lib_sort_key)
            lib_headers = []
            for name in ordered_libs:
                note = _LIB_WIDTH_NOTE.get(name)
                lib_headers.append(f"{name}({note})" if note else name)

            header = ["size", "vecrec*"] + lib_headers + ["vr\u00d7"]
            rows: list[list[str]] = []
            winners: list[str] = []
            any_flag = False
            for p in data.points:
                vs, ls = _bucket_entries(p, mode, width)
                best, flagged = _best_vecrec(vs)
                any_flag = any_flag or flagged
                by_name = {r.impl: r for r in ls}

                vr_cell = "-"
                if best is not None:
                    vr_cell = _fmt_time(_median(best)) + (" \u2717" if flagged else "")
                    winners.append(f"{p.size}\u2192{_short_sched(best.impl)}")

                lib_cells = []
                for name in ordered_libs:
                    r = by_name.get(name)
                    if r is None:
                        lib_cells.append("-")
                    elif r.note == "not built":
                        lib_cells.append("SKIP")
                    else:
                        cell = _fmt_time(_median(r))
                        if r.correct is False:
                            cell += " \u2717"
                        lib_cells.append(cell)

                best_lib = _best_lib_time(ls)
                spd = "-"
                if best is not None and best_lib is not None and _median(best) > 0:
                    spd = f"{best_lib / _median(best):.2f}\u00d7"
                rows.append([str(p.size), vr_cell] + lib_cells + [spd])

            footers = []
            if winners:
                footers.append("    vecrec* winner by size: " + "  ".join(winners))
            if not ordered_libs:
                footers.append("    (no library baseline at this width for this recurrence)")
            if any_flag:
                footers.append(
                    "    \u2717 = fastest vecrec schedule failed correctness (saturation/band); "
                    "shown time is fastest passing schedule (or fastest overall if none passed)"
                )
            mode_lines.extend(
                _render_table(
                    f"{width.upper()} ({mode})", header, rows, footers
                )
            )
            mode_lines.append("")

        if mode_has_any:
            lines.append("")
            lines.append(f"### {mode_title} ###")
            lines.extend(mode_lines)

    return lines


# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------


@dataclass
class CheckReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def run_checks(data: RecurrenceData) -> CheckReport:
    """Structural + correctness sanity checks over a recurrence's gathered data.

    Errors (gate the exit code):
      * a correctness-CHECKED library baseline that FAILS the reference -- this
        is a harness/reference sanity failure, exactly what the main suite gates
        on (VecRec entries never gate; their FAILs are expected data).
      * a baseline that errored out (crash / bad JSON).
      * an internal ``vr×`` vs raw-number inconsistency.

    Warnings (reported, non-fatal):
      * a width with a VecRec entry but zero passing schedules at every size.
      * a bucket where the fastest VecRec time is 0 (timing not captured).
    """
    rep = CheckReport()
    label = data.label

    for p in data.points:
        for r in p.results:
            # Errored baselines are always a problem.
            if r.note.startswith("error:"):
                rep.errors.append(
                    f"[{label} N={p.size}] baseline {r.impl} errored: {r.note}"
                )
                continue
            # A correctness-checked *library* baseline must validate. correct is
            # None when unchecked (timing-only) or when there's no reference.
            if not r.impl.startswith("vecrec") and r.correct is False:
                rep.errors.append(
                    f"[{label} N={p.size}] correctness-checked library {r.impl} "
                    f"FAILED vs reference (score={r.score})"
                )

    # Per-bucket structural checks + vr× consistency.
    for mode, _ in MODES:
        for width in WIDTHS:
            saw_vecrec = False
            saw_pass = False
            for p in data.points:
                vs, ls = _bucket_entries(p, mode, width)
                if not vs:
                    continue
                saw_vecrec = True
                best, _flag = _best_vecrec(vs)
                if best is not None and best.correct is not False:
                    saw_pass = True
                if best is not None and _median(best) <= 0:
                    rep.warnings.append(
                        f"[{label} {width} {mode} N={p.size}] vecrec best has no timing"
                    )
                # vr× consistency: recompute and compare to raw ratio.
                best_lib = _best_lib_time(ls)
                if best is not None and best_lib is not None and _median(best) > 0:
                    ratio = best_lib / _median(best)
                    if not (ratio > 0):
                        rep.errors.append(
                            f"[{label} {width} {mode} N={p.size}] vr× non-positive ({ratio})"
                        )
            if saw_vecrec and not saw_pass:
                rep.warnings.append(
                    f"[{label} {width} {mode}] no VecRec schedule passed correctness at any size "
                    "(expected for I8 once scores exceed its range)"
                )

    # Every rendered width must be one of the known keys (guards classify()).
    for p in data.points:
        for r in p.results:
            b = classify(r.impl)
            if b is not None:
                assert b.width in WIDTHS, f"bad width bucket {b.width} for {r.impl}"
                assert b.mode in ("matrix", "score"), f"bad mode {b.mode} for {r.impl}"

    return rep


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-recurrence width/mode VecRec-vs-library comparison matrix."
    )
    parser.add_argument(
        "--sizes",
        help="Comma-separated square sizes to override each recurrence's default sweep.",
    )
    parser.add_argument(
        "--recurrence",
        action="append",
        default=None,
        metavar="SUBSTR",
        help="Only run recurrences whose label contains SUBSTR (repeatable).",
    )
    args = parser.parse_args(argv)

    size_override: Optional[tuple[int, ...]] = None
    if args.sizes:
        size_override = tuple(int(x.strip()) for x in args.sizes.split(",") if x.strip())
        if not size_override:
            parser.error("--sizes must list at least one integer size")

    recs = _registry()
    if args.recurrence:
        subs = [s.lower() for s in args.recurrence]
        recs = [r for r in recs if any(s in r.label.lower() for s in subs)]
        if not recs:
            parser.error(f"no recurrence matched {args.recurrence}")

    all_errors: list[str] = []
    all_warnings: list[str] = []

    print("VecRec width/mode comparison matrix")
    print("raw = median wall time (lower is better); vr\u00d7 = best library time \u00f7 VecRec time")
    print("I8 vs I8-only libs; I32 vs I32-only libs; I16 vs I16 libs + SeqAn3/ksw2/edlib (internal width in parens)")

    for rec in recs:
        sizes = size_override or rec.default_sizes
        data = run_recurrence(rec, sizes)
        for line in render_recurrence(data):
            print(line)
        report = run_checks(data)
        all_errors.extend(report.errors)
        all_warnings.extend(report.warnings)

    print("")
    print("=" * 78)
    print("CHECKS")
    print("=" * 78)
    if all_warnings:
        print(f"  {len(all_warnings)} warning(s):")
        for w in all_warnings:
            print(f"    WARN  {w}")
    if all_errors:
        print(f"  {len(all_errors)} error(s):")
        for e in all_errors:
            print(f"    ERROR {e}")
        print("  RESULT: FAIL")
        return 1
    print("  all correctness + structural checks passed")
    print("  RESULT: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
