"""VecRec's 1-piece affine drop-in kernel vs ksw2 SIMD extension (``ksw_extz2_sse``).

This is the SAME recurrence as the real minimap2 drop-in kernel
(``end_to_end_baselines/scripts/generate_kernel.py``'s
``_build_minimap2_single_affine_expr``) -- a collapsed 2-state affine, reused
here from ``affine_gap._build_expr()``. This file is the controlled,
synthetic-data correctness/perf anchor for that shared recurrence.

``baselines/ksw2/align.cpp`` calls ``ksw_extz2_sse`` with ``flag=0`` (i.e.
*without* ``KSW_EZ_EXTZ_ONLY``) and ``zdrop=-1`` (disabled). Per
``ksw2_extz2_sse.c``'s own backtrack logic, that is "gap-fill" mode: it
backtracks from the corner ``(qlen, tlen)`` and ``ez.score`` IS the global
corner H(qlen,tlen) -- the same quantity ``ksw_gg2_sse`` and the textbook
affine DP compute. So, unlike a genuine ``KSW_EZ_EXTZ_ONLY`` call (free-end,
best-cell), this invocation is directly correctness-checkable against the
same reference used for the global-affine case: unbanded is checked here.

Banded is checked too, but is a genuinely tighter case: ksw2's diagonal-band
cutoff (``st > en -> zdropped=1``) can leave the corner unreached (and
``ez.score`` at its zero-initialized default) if the band can't cover it at
the final anti-diagonal; watch for spurious FAILs there distinct from
VecRec's own known banded diagonal-carry approximation (see
``banded_smith_waterman.py``).

Unbanded (full matrix) and banded (``w``) variants are swept.
"""

from __future__ import annotations

import numpy as np

from vecrec.expr.base import SignalExpr
from vecrec.util import ElementType

from evals_common.cpp import CppBaseline
from evals_common.harness import Baseline, EvalCase

from ._alignment import (
    SWEEP_SIZES,
    affine_reference,
    banded_affine_reference,
    sequence_inputs,
)
from .affine_gap import GE, GO, _build_expr

BAND_WIDTH = 128


class ExtensionAffine(EvalCase):
    name = "extension-affine"
    vecrec_bytes_per_cell = 4
    report_gcups = False
    boundary_convention = (
        "vecrec: TropMax collapsed 2-state affine (corner) -- the real minimap2 "
        "1-piece drop-in kernel; ksw2: ksw_extz2_sse SIMD 1-piece affine, called "
        "in gap-fill mode (flag=0) so its score is the same global corner"
    )

    def __init__(self, rows: int = 256, cols: int = 256, band_width: int = 0, seed: int = 0) -> None:
        self.rows = rows
        self.cols = cols
        self.band_width = band_width
        self.seed = seed
        self.n_elements = rows * cols

    def display_name(self) -> str:
        band = f" bw={self.band_width}" if self.band_width else " full"
        return f"{self.name} {self.rows}x{self.cols}{band}"

    @classmethod
    def sweep_variants(cls) -> list[ExtensionAffine]:
        variants: list[ExtensionAffine] = []
        for n in SWEEP_SIZES:
            variants.append(cls(rows=n, cols=n, band_width=0))
            variants.append(cls(rows=n, cols=n, band_width=BAND_WIDTH))
        return variants

    def make_inputs(self) -> dict[str, object]:
        return sequence_inputs(
            self.rows, self.cols, gap_open=-GO, gap_extend=-GE, seed=self.seed
        )

    def reference(self, **inputs: object) -> np.ndarray:
        if self.band_width:
            return banded_affine_reference(inputs["s"], GO, GE, self.band_width).flatten()
        return affine_reference(inputs["s"], GO, GE).flatten()

    def optimal_score(self, matrix: np.ndarray) -> float:
        return float(matrix[-1])

    def kernel_expr(self, element_type: ElementType = ElementType.I16) -> SignalExpr:
        return _build_expr(element_type)

    def baselines(self) -> list[Baseline]:
        return []

    def cpp_baselines(self) -> list:
        from evals_common.vecrec_cpp import vecrec_traceback_baselines_multi_width

        bw = self.band_width
        ksw_bw = bw if bw else -1
        return [
            CppBaseline(
                "ksw2_extz2_sse", "ksw2_align", "extz2_sse",
                extra_payload={"band_width": ksw_bw},
            ),
            # Isolated approx-max A/B: same kernel call, KSW_EZ_APPROX_MAX set
            # (minimap2's gap-fill pass-1 shortcut, MM_EXACT_ONLY=0 in the e2e
            # drop-in). Gap-fill mode (no KSW_EZ_EXTZ_ONLY) always backtracks
            # from the corner regardless of this flag, so ez.score is expected
            # bit-identical to the row above -- check_correctness stays on to
            # confirm that empirically at every sweep size, and this isolates
            # the flag's speed effect alone, with none of the minimap2 pipeline
            # overhead (indexing/chaining/SAM I/O) the e2e `approx-max`
            # scenario (end_to_end_baselines/minimap2/scenarios.py) carries.
            CppBaseline(
                "ksw2_extz2_sse_approx", "ksw2_align", "extz2_sse",
                extra_payload={"band_width": ksw_bw, "approx_max": 1},
            ),
            # I8/I16/I32 width sweep (not just I16) -- autoscheduling picks
            # (width, schedule), matching the real minimap2 drop-in's dual-width
            # I8/I16 dispatch (generate_kernel.py). I8 saturates (clamps, not
            # wraps) past a score-dependent N; FAIL there is expected and does
            # not gate the suite -- see cases/README.md "VecRec element-width sweep".
            *vecrec_traceback_baselines_multi_width(self.kernel_expr, kind="affine", band_width=bw),
        ]


if __name__ == "__main__":
    from evals_common.harness import run_cases

    raise SystemExit(run_cases([ExtensionAffine()]))
