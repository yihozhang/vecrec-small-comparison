"""Banded Needleman-Wunsch global sequence alignment.

vecrec kernel: the same TropMax 2D recurrence as NeedlemanWunsch, compiled
with ``band_width > 0`` so that each row's column sweep is restricted to a
symmetric diagonal band of half-width ``band_width``.

A warm-up pass runs ``kernel.run()`` from col=0 to ``col_start`` without
writing to the output array, correctly priming ProfileSignal2D row pointers,
RepeaterVec ring buffer, and SRecurse carry before the in-band loop.  For NW
with linear gap, cells outside the band have score -inf (TropMax zero), so the
warm-up carry at col_start is the true boundary value and scores are exact.

Baselines:
- SeqAn3 banded global NW (``nw_banded`` algo tag, ``band_fixed_size``).
  Scores should match VecRec exactly for alignments whose optimal path stays
  within the band.
- ksw2's ``ksw_gg2_sse`` (1-piece affine GLOBAL, banded, SIMD). With
  gap_open=0 it degenerates to linear gap, matching the reference exactly --
  a second, independent, vectorized correctness anchor.
- parasail's ``nw_banded`` (scalar, banded). A third independent correctness
  anchor; not meaningful for timing (no SIMD).
- ksw2's ``ksw_extz2_sse`` (1-piece affine EXTENSION, banded, SIMD) is the
  fairest *vectorized timing* comparison, but its extension semantics
  legitimately diverge from the global reference, so it is timing-only.
"""

from __future__ import annotations

import numpy as np

from evals_common.cpp import CppBaseline
from evals_common.harness import Baseline, EvalCase

from vecrec.util import ElementType

from ._alignment import banded_nw_reference, sequence_inputs
from .needleman_wunsch import GAP, _build_expr

BAND_WIDTH = 128  # default diagonal half-width


class BandedNeedlemanWunsch(EvalCase):
    name = "banded-needleman-wunsch"
    vecrec_bytes_per_cell = 4
    report_gcups = False
    boundary_convention = (
        "vecrec: TropMax banded global (I16); seqan3: banded global NW"
    )

    def __init__(
        self,
        rows: int = 128,
        cols: int = 128,
        band_width: int = BAND_WIDTH,
        seed: int = 0,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.band_width = band_width
        self.seed = seed
        self.n_elements = rows * cols

    def display_name(self) -> str:
        return f"{self.name} {self.rows}x{self.cols} bw={self.band_width}"

    @classmethod
    def sweep_variants(cls) -> list[BandedNeedlemanWunsch]:
        return [cls(rows=n, cols=n, band_width=BAND_WIDTH) for n in [128, 256, 512, 1024]]

    def make_inputs(self) -> dict[str, object]:
        return sequence_inputs(
            self.rows, self.cols, gap_open=0, gap_extend=-GAP, seed=self.seed
        )

    def reference(self, **inputs: object) -> np.ndarray:
        return banded_nw_reference(inputs["s"], GAP, self.band_width).flatten()

    def optimal_score(self, matrix: np.ndarray) -> float:
        return float(matrix[-1])

    def kernel_expr(self, element_type: ElementType = ElementType.I16):
        return _build_expr(element_type)

    def baselines(self) -> list[Baseline]:
        return [
            Baseline(
                "numpy_loop",
                lambda **kw: banded_nw_reference(
                    kw["s"], GAP, self.band_width
                ).flatten(),
            )
        ]

    def cpp_baselines(self) -> list:
        from evals_common.vecrec_cpp import (
            vecrec_traceback_baselines,
            vecrec_traceback_baselines_multi_width,
        )

        return [
            # I8 schedules use wider lanes (more elements per iter), so a fixed
            # band_width tuned for I16 may fall under the `band_width >= 4*epi`
            # rule (see root CLAUDE.md) for the widest I8 schedules -- those
            # will show correctness FAIL from band-rounding, not saturation.
            # Both failure modes are informative (and don't gate the suite for
            # vecrec entries) about which width is safe to use at this band.
            *vecrec_traceback_baselines_multi_width(
                self.kernel_expr, kind="nw", band_width=self.band_width
            ),
            CppBaseline(
                "seqan3_nw_banded",
                "seqan3_nw",
                "nw_banded",
                extra_payload={"band_width": self.band_width},
            ),
            # SIMD 1-piece affine GLOBAL, banded (native SSE2 on x86, sse2neon on
            # ARM). gap_open=0 (from make_inputs) degenerates it to pure linear
            # gap, matching banded_nw_reference exactly -- a second, independent
            # correctness anchor alongside SeqAn3 banded.
            CppBaseline(
                "ksw2_gg2_sse",
                "ksw2_align",
                "gg2_sse",
                extra_payload={"band_width": self.band_width},
            ),
            # Scalar parasail banded NW: a third, independent correctness anchor.
            # parasail's ``k`` is a symmetric diagonal half-width that equals
            # band_width exactly for equal-length sequences (see nw_banded.c);
            # this case always runs with rows == cols. Timing is not meaningful
            # here (scalar, no SIMD) -- this baseline is for correctness only.
            CppBaseline(
                "parasail_nw_banded",
                "parasail_align",
                "nw_banded",
                extra_payload={"band_width": self.band_width},
            ),
            # SIMD 1-piece affine extension (the vectorized ksw2 kernel family
            # minimap2 calls; native SSE2 on x86, sse2neon on ARM). With
            # gap_open=0 it runs the linear-gap recurrence at the same band
            # width, making it the fairest *vectorized* timing comparison for
            # banded NW. It uses extension (not global) semantics, so its score
            # legitimately won't match the banded-global reference — timing-only.
            CppBaseline(
                "ksw2_extz2_sse",
                "ksw2_align",
                "extz2_sse",
                check_correctness=False,
                extra_payload={"band_width": self.band_width},
            ),
            # SIMD linear-gap X-drop extension kernel (score-only, no CIGAR),
            # banded. Its banded geometry differs from the symmetric band this
            # case uses, so timing-only. Paired with a score-only VecRec driver
            # so memory footprint and timed span match this baseline's kernel.
            CppBaseline(
                "ksw2_extf2_sse",
                "ksw2_align",
                "extf2_sse",
                check_correctness=False,
                extra_payload={"band_width": self.band_width},
            ),
            *vecrec_traceback_baselines(
                self.kernel_expr(), kind="nw", band_width=self.band_width, score_only=True
            ),
        ]


if __name__ == "__main__":
    from evals_common.harness import run_cases

    raise SystemExit(run_cases([BandedNeedlemanWunsch()]))
