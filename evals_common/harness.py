"""Evaluation harness (suite-agnostic).

An :class:`EvalCase` bundles together a vecrec implementation (which generates,
compiles, and runs a C++ benchmark via ``vecrec.codegen.generate_and_run_benchmark``)
and zero or more baselines. Baselines come in two flavors:

* :class:`Baseline` -- a Python callable (numpy, scipy, ...) timed in-process.
* :class:`evals_common.cpp.CppBaseline` -- a compiled native driver under
  ``bin/`` that self-times and reports its result as JSON.

The runner executes the vecrec kernel and every baseline, checks correctness
(full-matrix ``allclose`` or, for score-only baselines, optimal-score equality),
and prints a unified throughput table (GCUPS + speedup vs best) with timing
statistics and deeper correctness metrics.
"""

from __future__ import annotations

import functools
import importlib
import pkgutil
import platform
import subprocess
import time
import traceback
from abc import ABC
from dataclasses import dataclass
from statistics import median
from types import ModuleType
from typing import Callable, Optional, Sequence

import numpy as np

from evals_common.metrics import (
    ImplResult,
    WorkMode,
    apply_speedup_vs_best,
    effective_seconds,
    gcups_from_mb_s,
    gcups_from_seconds,
    timing_stats,
)

ArrayFn = Callable[..., np.ndarray]


@functools.lru_cache(maxsize=1)
def platform_compiler_flags() -> list[str]:
    """Extra ``clang++`` flags needed to compile on the current platform.

    On macOS the Homebrew LLVM ``clang++`` that vecrec invokes by default does
    not find the system C/C++ SDK headers unless it is pointed at the active SDK
    with ``-isysroot``. We resolve the SDK path via ``xcrun`` and pass it
    through; on every other platform no extra flags are needed.

    Cases should forward the result as ``compiler_flags=`` to
    ``vecrec.codegen.generate_and_run_benchmark``.
    """
    if platform.system() != "Darwin":
        return []
    try:
        sdk_path = subprocess.run(
            ["xcrun", "--show-sdk-path"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return []
    if not sdk_path:
        return []
    return ["-isysroot", sdk_path]


def run_auto_scheduled(expr, codegen=None, **benchmark_kwargs) -> dict:
    """Auto-schedule ``expr`` and run the vecrec benchmark across the candidates.

    A convenience wrapper for :meth:`EvalCase.run_vecrec` implementations that
    don't want to hand-write a schedule: it enumerates candidate schedules with
    ``vecrec.optimize.get_candidate_schedules`` and benchmarks them all (the
    first kernel doubles as the cross-schedule correctness reference). The
    platform-specific compiler flags from :func:`platform_compiler_flags` are
    applied automatically so cases never have to special-case macOS.

    Returns the dict produced by
    ``vecrec.codegen.generate_and_run_benchmark``.
    """
    from vecrec.codegen import CodeGen, generate_and_run_benchmark
    from vecrec.optimize import get_candidate_schedules
    from vecrec.util import get_native_bits

    # vecrec's get_native_bits() shells out to `lscpu`, which doesn't exist on
    # macOS and raises FileNotFoundError (it only guards CalledProcessError).
    # Fall back to its own 256-bit default in that case.
    try:
        max_bits = get_native_bits()
    except OSError:
        max_bits = 256

    schedules = get_candidate_schedules(expr, max_bits)
    return generate_and_run_benchmark(
        codegen or CodeGen(),
        schedules,
        include_correctness_check=True,
        compiler_flags=platform_compiler_flags(),
        **benchmark_kwargs,
    )


@dataclass
class Baseline:
    """A reference implementation from an external Python library.

    ``fn`` receives the case inputs as keyword arguments (the dict returned by
    :meth:`EvalCase.make_inputs`) and must return the output array.
    """

    name: str
    fn: ArrayFn


@dataclass
class TimedOutput:
    seconds: float
    output: object
    samples: list[float]


class EvalCase(ABC):
    """A single comparative evaluation.

    Subclasses implement :meth:`run_vecrec` and may override :meth:`make_inputs`,
    :meth:`baselines`, :meth:`cpp_baselines`, :meth:`reference`, and
    :meth:`optimal_score` to enable baseline timing and correctness checking.
    """

    #: Human-readable name shown in the results table.
    name: str = "unnamed-case"

    #: Number of input elements processed; used to report throughput for
    #: baselines. For 2D cases this is the DP-matrix cell count (rows*cols), which
    #: also makes the reported GCUPS meaningful. Override per case.
    n_elements: int = 1 << 20

    #: Bytes read+written per DP cell in the vecrec benchmark (input + output).
    #: For 2D I64 kernels with one ``s`` input and one I64 output this is 16.
    vecrec_bytes_per_cell: int = 8

    #: How vecrec reports work relative to score-only C++ baselines.
    vecrec_work_mode: WorkMode = "full_matrix"

    #: Work mode tag for Python full-output baselines (numpy, scipy, ...).
    baseline_work_mode: WorkMode = "full_matrix"

    #: Tag describing boundary/scoring conventions (shown in correctness section).
    boundary_convention: str = ""

    #: When False, throughput table uses wall time only (bio real-input cases).
    report_gcups: bool = True

    matrix_rtol: float = 1e-4
    matrix_atol: float = 1e-4
    score_atol: float = 1e-6

    def run_vecrec(self) -> dict:
        """Build, transform, compile, and run the vecrec random-input benchmark.

        Override for leetcode-style cases. Bio alignment cases use
        :meth:`kernel_expr` instead; the default raises if neither is provided.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement run_vecrec or kernel_expr"
        )

    def display_name(self) -> str:
        """Case label including sweep parameters when present."""
        rows = getattr(self, "rows", None)
        cols = getattr(self, "cols", None)
        seed = getattr(self, "seed", None)
        if rows is not None and cols is not None:
            suffix = f" ({rows}x{cols}"
            if seed is not None:
                suffix += f", seed={seed}"
            suffix += ")"
            return f"{self.name}{suffix}"
        return self.name

    def sweep_params(self) -> dict[str, object]:
        """Metadata describing the problem instance (for result artifacts)."""
        params: dict[str, object] = {"cells": self.n_elements}
        for key in ("rows", "cols", "seed", "gap_open", "gap_extend"):
            if hasattr(self, key):
                params[key] = getattr(self, key)
        return params

    @classmethod
    def sweep_variants(cls) -> Sequence[EvalCase]:
        """Optional size sweep. Override in alignment cases."""
        return (cls(),)

    def make_inputs(self) -> dict[str, object]:
        """Inputs passed as kwargs to each baseline and to :meth:`reference`."""
        return {}

    def baselines(self) -> Sequence[Baseline]:
        """External Python-library reference implementations to time/validate."""
        return ()

    def cpp_baselines(self) -> Sequence["object"]:
        """Native C++ driver baselines (``evals_common.cpp.CppBaseline``).

        Returned lazily so importing a case never requires the native drivers to
        be built; unbuilt drivers are reported and skipped at run time.
        """
        return ()

    def reference(self, **inputs: object) -> Optional[np.ndarray]:
        """Ground-truth output used to validate baselines.

        Return ``None`` to skip correctness checking.
        """
        return None

    def optimal_score(self, matrix: np.ndarray) -> Optional[float]:
        """Reduce a full reference matrix to a single optimal score.

        Override in cases whose C++ (or score-only) baselines report a scalar
        optimal alignment score rather than the full DP matrix. Returning
        ``None`` (default) disables score-level comparison.
        """
        return None

    def kernel_expr(self) -> Optional[object]:
        """Return the (unscheduled) vecrec expression for this case's kernel.

        When non-None, the harness treats this as a bio alignment case: vecrec is
        measured by the self-timed C++ traceback driver(s) returned from
        :meth:`cpp_baselines` (see :mod:`evals_common.vecrec_cpp`), and the
        in-Python ``run_vecrec`` benchmark path is skipped. Leave as None for
        random-input benchmark cases (which implement :meth:`run_vecrec`).
        """
        return None


def time_callable(
    fn: ArrayFn,
    inputs: dict[str, object],
    *,
    warmup: int = 2,
    iterations: int = 5,
) -> TimedOutput:
    """Return median seconds, last output, and raw samples for ``fn(**inputs)``."""
    for _ in range(warmup):
        fn(**inputs)
    samples: list[float] = []
    output: object = None
    for _ in range(iterations):
        start = time.perf_counter()
        output = fn(**inputs)
        samples.append(time.perf_counter() - start)
    return TimedOutput(median(samples), output, samples)


def _matrix_match_pct(
    output: object,
    expected: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> float:
    out = np.asarray(output).reshape(expected.shape)
    if expected.size == 0:
        return 100.0
    close = np.isclose(out, expected, rtol=rtol, atol=atol)
    return float(close.mean()) * 100.0


def _score_check(
    score: float,
    expected: float,
    *,
    atol: float,
) -> tuple[bool, float]:
    delta = abs(float(score) - float(expected))
    return bool(np.isclose(float(score), float(expected), rtol=0, atol=atol)), delta


def _baseline_failure(exc: BaseException) -> tuple[str, str]:
    """Return a short summary and full traceback for a baseline error."""
    tb = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    ).rstrip()
    summary = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    return f"error: {summary}", tb


def _collect_vecrec_results(
    case: EvalCase,
    vecrec_output: dict,
    *,
    expected_score: Optional[float],
) -> list[ImplResult]:
    output = vecrec_output.get("output", {})
    if not isinstance(output, dict):
        return []

    validation = output.get("validation")
    all_valid = None
    if isinstance(validation, dict) and validation:
        all_valid = all(v is True for v in validation.values())

    timing_by_kernel = vecrec_output.get("timing", {})
    if not isinstance(timing_by_kernel, dict):
        timing_by_kernel = {}

    results: list[ImplResult] = []
    for kernel, value in output.items():
        if kernel == "validation":
            continue
        mb_s = float(value)
        gcups = gcups_from_mb_s(mb_s, case.vecrec_bytes_per_cell)

        kernel_timing = timing_by_kernel.get(kernel, {})
        samples: list[float] = []
        if isinstance(kernel_timing, dict):
            raw = kernel_timing.get("samples", [])
            if isinstance(raw, list):
                samples = [float(s) for s in raw]

        stats = timing_stats(samples) if samples else None
        seconds = stats.median if stats is not None else 0.0
        if seconds <= 0 and gcups is not None and gcups > 0:
            seconds = case.n_elements / gcups / 1e9

        results.append(
            ImplResult(
                impl=f"vecrec:{kernel}",
                work_mode=case.vecrec_work_mode,
                cells=case.n_elements,
                seconds=seconds,
                gcups=gcups,
                correct=all_valid,
                timing=stats,
                note="internal cross-schedule validation",
                boundary_convention=case.boundary_convention,
            )
        )
    return results


def _run_baselines(case: EvalCase) -> list[ImplResult]:
    baselines = list(case.baselines())
    cpp_baselines = list(case.cpp_baselines())
    if not baselines and not cpp_baselines:
        return []

    inputs = case.make_inputs()
    expected_matrix = case.reference(**inputs)
    expected_score = (
        case.optimal_score(expected_matrix) if expected_matrix is not None else None
    )

    results: list[ImplResult] = []

    for baseline in baselines:
        try:
            timed = time_callable(baseline.fn, inputs)
        except Exception as exc:  # noqa: BLE001 - report and continue
            note, tb = _baseline_failure(exc)
            results.append(
                ImplResult(
                    impl=baseline.name,
                    work_mode=case.baseline_work_mode,
                    cells=case.n_elements,
                    seconds=0.0,
                    gcups=None,
                    correct=False,
                    note=note,
                    traceback=tb,
                )
            )
            continue

        gcups = (
            None
            if not case.report_gcups
            else gcups_from_seconds(case.n_elements, timed.seconds)
        )
        matrix_match: Optional[float] = None
        correct: Optional[bool] = None
        score: Optional[float] = None
        score_delta: Optional[float] = None

        if expected_matrix is not None:
            matrix_match = _matrix_match_pct(
                timed.output,
                expected_matrix,
                rtol=case.matrix_rtol,
                atol=case.matrix_atol,
            )
            correct = matrix_match >= 100.0 - 1e-9
            if expected_score is not None:
                out_arr = np.asarray(timed.output).reshape(expected_matrix.shape)
                score = float(case.optimal_score(out_arr))
                ok, score_delta = _score_check(
                    score, expected_score, atol=case.score_atol
                )
                correct = correct and ok

        results.append(
            ImplResult(
                impl=baseline.name,
                work_mode=case.baseline_work_mode,
                cells=case.n_elements,
                seconds=timed.seconds,
                gcups=gcups,
                correct=correct,
                score=score,
                score_delta=score_delta,
                matrix_match_pct=matrix_match,
                timing=timing_stats(timed.samples),
                boundary_convention="textbook reference",
            )
        )

    for cpp in cpp_baselines:
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
            note, tb = _baseline_failure(exc)
            results.append(
                ImplResult(
                    impl=cpp.name,
                    work_mode="traceback",
                    cells=case.n_elements,
                    seconds=0.0,
                    gcups=None,
                    correct=False,
                    note=note,
                    traceback=tb,
                )
            )
            continue

        stats = timing_stats(res.samples) if res.samples else timing_stats([res.seconds])
        correct: Optional[bool] = None
        score_delta: Optional[float] = None
        if expected_score is not None and getattr(cpp, "check_correctness", True):
            correct, score_delta = _score_check(
                float(res.score), expected_score, atol=case.score_atol
            )

        align_query = ""
        align_ref = ""
        align_comp = ""
        if res.alignment is not None:
            align_query = res.alignment.query
            align_ref = res.alignment.ref
            align_comp = res.alignment.comp

        cpp_gcups = None if not case.report_gcups else res.gcups
        results.append(
            ImplResult(
                impl=cpp.name,
                work_mode="traceback",
                cells=res.cells,
                seconds=res.seconds,
                gcups=cpp_gcups,
                correct=correct,
                score=float(res.score),
                score_delta=score_delta,
                timing=stats,
                note="score+alignment",
                boundary_convention="textbook reference (C++ drivers)",
                align_query=align_query,
                align_ref=align_ref,
                align_comp=align_comp,
            )
        )

    return results


def _format_correctness(r: ImplResult) -> str:
    if r.note == "not built":
        return "SKIP"
    if r.correct is None:
        return "-"
    if not r.correct:
        return "FAIL"
    return "PASS"


def _truncate_alignment(text: str, *, limit: int = 96) -> str:
    if len(text) <= limit:
        return text
    half = (limit - 3) // 2
    return f"{text[:half]}...{text[-half:]}"


def _format_score_info(r: ImplResult) -> str:
    parts: list[str] = []
    if r.score is not None:
        parts.append(f"score={r.score:.0f}" if r.score == int(r.score) else f"score={r.score}")
    if r.score_delta is not None:
        parts.append(f"delta={r.score_delta:g}")
    if r.matrix_match_pct is not None:
        parts.append(f"matrix={r.matrix_match_pct:.1f}%")
    if r.note == "internal cross-schedule validation":
        parts.append("internal")
    elif r.note.startswith("error:"):
        parts.append(r.note)
    return "  ".join(parts) if parts else "-"


def _effective_seconds(r: ImplResult) -> float:
    return effective_seconds(r)


def _format_seconds(r: ImplResult) -> str:
    sec = _effective_seconds(r)
    if sec <= 0:
        return "-"
    if sec < 1e-5:
        return f"{sec * 1e6:.2f}µs"
    if sec < 1e-3:
        return f"{sec * 1e6:.1f}µs"
    if sec < 1.0:
        return f"{sec * 1e3:.3f}ms"
    return f"{sec:.4f}s"


def _format_p10_p90(r: ImplResult) -> str:
    if r.timing is None or r.timing.p10 is None or r.timing.p90 is None:
        return "-"
    return f"{_format_seconds_short(r.timing.p10)}-{_format_seconds_short(r.timing.p90)}"


def _format_seconds_short(sec: float) -> str:
    if sec < 1e-5:
        return f"{sec * 1e6:.1f}µs"
    if sec < 1e-3:
        return f"{sec * 1e6:.0f}µs"
    if sec < 1.0:
        return f"{sec * 1e3:.1f}ms"
    return f"{sec:.3f}s"


def _format_cv(r: ImplResult) -> str:
    if r.timing is None or r.timing.cv_pct is None:
        return "-"
    return f"{r.timing.cv_pct:.1f}"


def _format_n_samples(r: ImplResult) -> str:
    if r.timing is None or not r.timing.samples:
        return "-"
    return str(len(r.timing.samples))


def _print_case_results(
    case: EvalCase,
    vecrec_output: dict,
    all_results: list[ImplResult],
    *,
    expected_score: Optional[float],
) -> None:
    print(f"\n=== {case.display_name()} ===")

    if case.boundary_convention:
        print(f"  convention: {case.boundary_convention}")
    if expected_score is not None:
        print(f"  reference score: {expected_score:g}")

    validation = None
    output = vecrec_output.get("output", {})
    if isinstance(output, dict):
        validation = output.get("validation")
    if validation is not None:
        print(f"  vecrec validation: {validation}")

    print("  correctness:")
    print(f"    {'impl':<28} {'result':>6}  details")
    for r in all_results:
        print(f"    {r.impl:<28} {_format_correctness(r):>6}  {_format_score_info(r)}")

    tracebacks = [(r.impl, r.traceback) for r in all_results if r.traceback]
    if tracebacks:
        print("  baseline tracebacks:")
        for impl, tb in tracebacks:
            print(f"    --- {impl} ---")
            for line in tb.splitlines():
                print(f"    {line}")

    alignments = [
        r
        for r in all_results
        if r.align_query or r.align_ref or r.align_comp
    ]
    if alignments:
        print("  alignments:")
        for r in alignments:
            print(f"    --- {r.impl} ---")
            print(f"      query: {_truncate_alignment(r.align_query)}")
            print(f"      ref:   {_truncate_alignment(r.align_ref)}")
            print(f"      comp:  {_truncate_alignment(r.align_comp)}")

    show_gcups = case.report_gcups
    if show_gcups:
        print("  throughput (GCUPS, higher is better):")
        print(
            f"    {'impl':<28} {'mode':<14} {'median':>10} {'GCUPS':>10} "
            f"{'vs_best':>8} {'p10-p90':>14} {'CV%':>6} {'n':>3}  correct"
        )
    else:
        print("  throughput (wall time, lower is better):")
        print(
            f"    {'impl':<28} {'mode':<14} {'median':>10} "
            f"{'vs_best':>8} {'p10-p90':>14} {'CV%':>6} {'n':>3}  correct"
        )
    for r in all_results:
        gc = f"{r.gcups:.3f}" if show_gcups and r.gcups is not None else "-"
        vs = f"{r.speedup_vs_best:.2f}x" if r.speedup_vs_best is not None else "-"
        if show_gcups:
            print(
                f"    {r.impl:<28} {r.work_mode:<14} {_format_seconds(r):>10} {gc:>10} "
                f"{vs:>8} {_format_p10_p90(r):>14} {_format_cv(r):>6} "
                f"{_format_n_samples(r):>3}  {_format_correctness(r)}"
            )
        else:
            print(
                f"    {r.impl:<28} {r.work_mode:<14} {_format_seconds(r):>10} "
                f"{vs:>8} {_format_p10_p90(r):>14} {_format_cv(r):>6} "
                f"{_format_n_samples(r):>3}  {_format_correctness(r)}"
            )

    vecrec_ks = [
        r for r in all_results
        if r.impl.startswith("vecrec:k") and r.impl[len("vecrec:k"):].isdigit()
    ]
    if show_gcups and len(vecrec_ks) >= 2:
        k0 = next((r for r in vecrec_ks if r.impl == "vecrec:k0"), None)
        best_k = max(vecrec_ks, key=lambda r: r.gcups or 0.0)
        if k0 is not None and best_k.impl != "vecrec:k0" and k0.gcups and best_k.gcups:
            speedup = best_k.gcups / k0.gcups
            print(f"  vectorization speedup k0 → {best_k.impl.split(':')[1]}: {speedup:.1f}x")

    # Highlight the fastest vecrec C++ traceback schedule (vecrec_<kind>_tb:k*).
    # Prefer schedules that passed correctness: a narrower element width (I8)
    # can be the fastest by time yet score wrong (saturated/clamped, or -- for
    # TropMin cases -- collided with too-small an infinity sentinel), so skip
    # those in favor of the next-best I16/I32 schedule that's actually
    # correct. Only fall back to the fastest-regardless-of-correctness if
    # nothing passed at all (so the line still reports something, clearly
    # flagged, rather than going silent).
    vecrec_tb = [
        r for r in all_results
        if r.impl.startswith("vecrec_") and ":k" in r.impl and _effective_seconds(r) > 0
    ]
    if len(vecrec_tb) >= 2:
        vecrec_tb_correct = [r for r in vecrec_tb if r.correct is not False]
        fastest_overall = min(vecrec_tb, key=_effective_seconds)
        if vecrec_tb_correct:
            best = min(vecrec_tb_correct, key=_effective_seconds)
            print(f"  fastest vecrec schedule: {best.impl} ({_format_seconds(best)})")
            if fastest_overall.impl != best.impl:
                print(
                    f"    ({fastest_overall.impl} was faster "
                    f"({_format_seconds(fastest_overall)}) but failed correctness -- skipped)"
                )
        else:
            print(
                f"  fastest vecrec schedule: {fastest_overall.impl} "
                f"({_format_seconds(fastest_overall)}) "
                "[WARNING: failed correctness -- no passing schedule at any width]"
            )


ResultSink = Callable[[EvalCase, list[ImplResult]], None]


def run_case(case: EvalCase, on_results: Optional[ResultSink] = None) -> bool:
    """Run a single case; return ``True`` if it succeeded (and baselines passed).

    When ``on_results`` is provided it is invoked as ``on_results(case, results)``
    with the full list of :class:`ImplResult` rows (speedups already filled in),
    letting callers persist results (e.g. to CSV) without re-running anything.
    """
    inputs = case.make_inputs()
    expected_matrix = case.reference(**inputs)
    expected_score = (
        case.optimal_score(expected_matrix) if expected_matrix is not None else None
    )

    if case.kernel_expr() is not None:
        # Bio alignment cases: vecrec is measured by the self-timed C++ traceback
        # driver(s) returned from ``cpp_baselines()`` (see evals_common.vecrec_cpp),
        # not by any in-Python timing.
        vecrec_results: list[ImplResult] = []
        vecrec_output: dict = {}
    else:
        vecrec_output = case.run_vecrec()
        assert (
            vecrec_output["return_code"] == 0
        ), f"vecrec benchmark failed for {case.name}: {vecrec_output}"
        vecrec_results = _collect_vecrec_results(
            case, vecrec_output, expected_score=expected_score
        )

    baseline_results = _run_baselines(case)
    all_results = vecrec_results + baseline_results

    apply_speedup_vs_best(all_results)
    if on_results is not None:
        on_results(case, all_results)
    _print_case_results(
        case, vecrec_output, all_results, expected_score=expected_score
    )

    # Exit status reflects whether the *trusted* baselines validate the reference
    # (a harness/reference sanity check). The vecrec drivers are the experimental
    # subject under test, not trusted baselines: their correctness is reported in
    # the table but does not gate the suite (e.g. a kernel whose boundary
    # convention diverges from the textbook reference is data, not a failure).
    return all(
        r.correct is not False
        for r in baseline_results
        if not r.impl.startswith("vecrec")
    )


def run_cases(cases: Sequence[EvalCase], on_results: Optional[ResultSink] = None) -> int:
    """Run all cases. Return a process exit code (0 on success).

    ``on_results`` is forwarded to :func:`run_case` (see there); it is called
    once per case that runs without raising.
    """
    ok = True
    for case in cases:
        try:
            ok = run_case(case, on_results=on_results) and ok
        except Exception as exc:  # noqa: BLE001 - report and continue
            ok = False
            print(f"\n=== {case.display_name()} ===\n  ERROR: {exc}")
            tb = traceback.format_exc().rstrip()
            if tb:
                print("  traceback:")
                for line in tb.splitlines():
                    print(f"    {line}")
    return 0 if ok else 1


def discover_cases(cases_pkg: ModuleType, *, sweep: bool = False) -> list[EvalCase]:
    """Instantiate every :class:`EvalCase` subclass found in ``cases_pkg``.

    When ``sweep`` is true, cases that define :meth:`EvalCase.sweep_variants`
    are expanded into multiple problem sizes.
    """
    found: list[EvalCase] = []
    for module_info in pkgutil.iter_modules(cases_pkg.__path__):
        module = importlib.import_module(f"{cases_pkg.__name__}.{module_info.name}")
        for attr in vars(module).values():
            if (
                isinstance(attr, type)
                and issubclass(attr, EvalCase)
                and attr is not EvalCase
            ):
                if sweep:
                    found.extend(attr.sweep_variants())
                else:
                    found.append(attr())
    return found


def run_suite(
    cases_pkg: ModuleType,
    *,
    sweep: bool = False,
    on_results: Optional[ResultSink] = None,
) -> int:
    """Discover and run every case in ``cases_pkg``. Return a process exit code."""
    cases = discover_cases(cases_pkg, sweep=sweep)
    if not cases:
        print(f"No evaluation cases found under {cases_pkg.__name__}.")
        return 0
    return run_cases(cases, on_results=on_results)
