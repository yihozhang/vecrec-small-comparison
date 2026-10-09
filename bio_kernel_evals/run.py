"""Entry point for the bio-kernel evaluation suite (``bio-evals``)."""

from __future__ import annotations

import argparse
import sys


def _alignment_sweep_cases(sizes: tuple[int, ...]) -> list:
    from bio_kernel_evals.cases._alignment import alignment_sweep_variants, banded_sweep_variants
    from bio_kernel_evals.cases.affine_gap import AffineGap, AffineGapSim65
    from bio_kernel_evals.cases.banded_affine import BandedAffine, BAND_WIDTH as BW_AFFINE
    from bio_kernel_evals.cases.banded_needleman_wunsch import (
        BandedNeedlemanWunsch,
        BAND_WIDTH as BW_NW,
    )
    from bio_kernel_evals.cases.banded_smith_waterman import (
        BandedSmithWaterman,
        BAND_WIDTH as BW_SW,
    )
    from bio_kernel_evals.cases.edit_distance import EditDistance, EditDistanceSim65
    from bio_kernel_evals.cases.needleman_wunsch import NeedlemanWunsch, NeedlemanWunschSim65
    from bio_kernel_evals.cases.smith_waterman import SmithWaterman
    from bio_kernel_evals.cases.affine_gap_local import AffineGapLocal
    from bio_kernel_evals.cases.extension_affine import (
        ExtensionAffine, BAND_WIDTH as BW_EXTZ,
    )
    from bio_kernel_evals.cases.extension_dual_affine import (
        ExtensionDualAffine, BAND_WIDTH as BW_EXTD,
    )

    cases: list = []
    for cls in (NeedlemanWunsch, SmithWaterman, AffineGap, EditDistance, AffineGapLocal):
        cases.extend(alignment_sweep_variants(cls, sizes=sizes))
    # WFA2-only timing variants on ~65% identity pairs (y = mutated copy of x).
    for cls in (NeedlemanWunschSim65, AffineGapSim65, EditDistanceSim65):
        cases.extend(alignment_sweep_variants(cls, sizes=sizes))
    for cls, bw in (
        (BandedNeedlemanWunsch, BW_NW),
        (BandedSmithWaterman, BW_SW),
        (BandedAffine, BW_AFFINE),
    ):
        cases.extend(banded_sweep_variants(cls, band_width=bw, sizes=sizes))
    # Extension-kernel VecRec-vs-ksw2 comparison suite (recurrence-matched,
    # SIMD-only) -- the controlled correctness/perf anchors for the same
    # kernels used in the real minimap2 drop-in.
    for cls in (ExtensionAffine, ExtensionDualAffine):
        cases.extend(alignment_sweep_variants(cls, sizes=sizes))
    for cls, bw in (
        (ExtensionAffine, BW_EXTZ),
        (ExtensionDualAffine, BW_EXTD),
    ):
        cases.extend(banded_sweep_variants(cls, band_width=bw, sizes=sizes))
    return cases


def main() -> None:
    from evals_common.harness import run_cases, run_suite

    import bio_kernel_evals.cases as cases

    parser = argparse.ArgumentParser(description="Run bio-kernel evaluation suite.")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep alignment cases over 64..1024 (64, 128, 256, 512, 1024).",
    )
    parser.add_argument(
        "--sweep-large",
        action="store_true",
        help="Sweep alignment cases over 1024..10000 (1024, 2048, 4096, 8192, 10000).",
    )
    parser.add_argument(
        "--sweep-sizes",
        metavar="SIZES",
        help="Sweep alignment cases at comma-separated square sizes (e.g. 1024,2048,4096).",
    )
    parser.add_argument(
        "--csv",
        nargs="?",
        const="results.csv",
        default=None,
        metavar="PATH",
        help=(
            "Save per-implementation results to a tidy CSV "
            "(default path: results.csv). Feed the file to "
            "bio_kernel_evals/make_table.py to build a LaTeX comparison table."
        ),
    )
    args = parser.parse_args()

    def execute(on_results) -> int:
        if args.sweep_large or args.sweep_sizes is not None:
            from bio_kernel_evals.cases._alignment import LARGE_SWEEP_SIZES

            if args.sweep_sizes is not None:
                sizes = tuple(
                    int(x.strip()) for x in args.sweep_sizes.split(",") if x.strip()
                )
                if not sizes:
                    parser.error("--sweep-sizes must list at least one size")
            else:
                sizes = LARGE_SWEEP_SIZES
            return run_cases(_alignment_sweep_cases(sizes), on_results=on_results)
        return run_suite(cases, sweep=args.sweep, on_results=on_results)

    if args.csv:
        from evals_common.results_csv import ResultsCsvWriter

        with ResultsCsvWriter(args.csv) as writer:
            code = execute(writer.write_case)
            print(f"\nWrote results to {args.csv}", file=sys.stderr)
            sys.exit(code)

    sys.exit(execute(None))


if __name__ == "__main__":
    main()
