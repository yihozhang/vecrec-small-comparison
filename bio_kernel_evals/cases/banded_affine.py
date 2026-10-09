"""Banded affine-gap (Gotoh) global sequence alignment.

vecrec kernel: the same collapsed 2-state affine recurrence as AffineGap,
compiled with ``band_width > 0`` so that each row's column sweep is
restricted to a symmetric diagonal band of half-width ``band_width``.

A warm-up pass runs ``kernel.run()`` from col=0 to ``col_start`` without
writing to the output array, correctly priming ProfileSignal2D row pointers,
RepeaterVec ring buffer, and SRecurse carry before the in-band loop.

Baselines:
- SeqAn3 banded global NW (``nw_banded`` algo tag) with real affine gap costs
  -- the correctness anchor; it matches ``banded_affine_reference`` exactly at
  every sweep size.
- ksw2's ``ksw_gg2_sse`` (1-piece affine GLOBAL, banded) -- timing-only. Its
  diagonal-band cutoff diverges from this case's symmetric-band reference once
  the optimum leaves the band (empirically the corner score disagrees from
  N>=256), so it is not correctness-checked (same treatment as ``ksw_extd2_sse``
  below); it remains a fair SIMD *timing* comparison (same DP loop + band width).
- ksw2's ``ksw_extd2_sse`` (2-piece affine, banded, extension semantics) --
  timing-only; its extension semantics legitimately diverge from the global
  reference near alignment boundaries, so it is not correctness-checked.
"""

from __future__ import annotations

import numpy as np

from vecrec.util import ElementType

from evals_common.cpp import CppBaseline
from evals_common.harness import Baseline, EvalCase

from ._alignment import banded_affine_reference, sequence_inputs
from .affine_gap import GO, GE, _build_expr

BAND_WIDTH = 128  # default diagonal half-width


class BandedAffine(EvalCase):
    name = "banded-affine"
    vecrec_bytes_per_cell = 4
    report_gcups = False
    boundary_convention = (
        "vecrec: TropMax banded collapsed 2-state affine (I16); "
        "ksw2_extd_scalar: scalar 2-piece affine banded (global semantics)"
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
    def sweep_variants(cls) -> list[BandedAffine]:
        return [cls(rows=n, cols=n, band_width=BAND_WIDTH) for n in [128, 256, 512, 1024]]

    def make_inputs(self) -> dict[str, object]:
        return sequence_inputs(
            self.rows, self.cols, gap_open=-GO, gap_extend=-GE, seed=self.seed
        )

    def reference(self, **inputs: object) -> np.ndarray:
        return banded_affine_reference(inputs["s"], GO, GE, self.band_width).flatten()

    def optimal_score(self, matrix: np.ndarray) -> float:
        return float(matrix[-1])

    def kernel_expr(self, element_type: ElementType = ElementType.I16):
        return _build_expr(element_type)

    def baselines(self) -> list[Baseline]:
        return [
            Baseline(
                "numpy_loop",
                lambda **kw: banded_affine_reference(
                    kw["s"], GO, GE, self.band_width
                ).flatten(),
            )
        ]

    def cpp_baselines(self) -> list:
        from evals_common.vecrec_cpp import vecrec_traceback_baselines_multi_width

        gap2_payload = {
            "band_width":  self.band_width,
            "gap_open2":   24,
            "gap_extend2": 1,
        }
        return [
            *vecrec_traceback_baselines_multi_width(
                self.kernel_expr, kind="affine", band_width=self.band_width
            ),
            # SIMD 1-piece affine GLOBAL, banded (native SSE2 on x86, sse2neon on
            # ARM). Same recurrence as the VecRec kernel, but ksw2's diagonal-band
            # cutoff diverges from this case's symmetric-band reference once the
            # optimum leaves the band -- empirically the banded corner score
            # disagrees from N>=256 (SeqAn3's banded driver stays exact). So this
            # is timing-only (check_correctness=False), like ksw2_extd2_sse below;
            # the DP loop + band width match, making it a fair SIMD timing anchor.
            CppBaseline(
                "ksw2_gg2_sse",
                "ksw2_align",
                "gg2_sse",
                check_correctness=False,
                extra_payload={"band_width": self.band_width},
            ),
            # SeqAn3 banded global NW driver reused with real affine gap costs
            # (its "nw_banded" tag applies whatever gap_open/gap_extend is in the
            # input JSON) -- a second independent correctness anchor.
            CppBaseline(
                "seqan3_nw_banded_affine",
                "seqan3_nw",
                "nw_banded",
                extra_payload={"band_width": self.band_width},
            ),
            # SIMD 2-piece affine (native SSE2 on x86, sse2neon on ARM). Uses
            # extension semantics (no trailing gap penalties at sequence ends), so
            # scores differ from VecRec's global alignment — but the DP loop and
            # band width are the same, making this the fairest SIMD timing
            # comparison. Its score legitimately won't match the banded-global
            # reference, so it is a timing-only baseline (check_correctness=False).
            CppBaseline(
                "ksw2_extd2_sse",
                "ksw2_align",
                "extd2_sse",
                check_correctness=False,
                extra_payload=gap2_payload,
            ),
        ]


if __name__ == "__main__":
    from evals_common.harness import run_cases

    raise SystemExit(run_cases([BandedAffine()]))
