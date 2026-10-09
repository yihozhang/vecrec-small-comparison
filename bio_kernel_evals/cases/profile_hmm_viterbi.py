"""Profile-HMM Viterbi (HMM lattice DP) -- vecrec vs vendored HMMER.

The non-alignment bio-eval: a profile-HMM Viterbi recurrence (coupled M/I/D states
with position-varying transitions = "affine-gap on a profile"). Both the vecrec
kernel and HMMER's p7_GViterbi run on the *same* model + observation and self-time in
C++ (model materialization + recurrence + traceback), exactly like the alignment
cases. HMMER (``baselines/hmmer/``) is the single source of the model: its
``hmm_prepare`` mode samples a seeded model, configures it UNIGLOCAL, zeros the
special-state ribbon, and dumps the exact configured scores it reads -- so numpy and
vecrec compute byte-for-byte the same DP (see ``_hmm.py`` / ``_hmm_kernel.py``).

Scores are float Viterbi log-odds (nats); vecrec, numpy, and HMMER agree within
float rounding. The optimum is the corner H[L][M] = max(M[L][M], D[L][M]).
"""

from __future__ import annotations

import numpy as np

from vecrec.expr.base import SignalExpr

from evals_common.harness import Baseline, EvalCase

from ._hmm import DEFAULT_SEED, hmm_inputs, hmm_sweep_variants, hmm_viterbi_reference
from ._hmm_kernel import build_expr


class ProfileHmmViterbi(EvalCase):
    name = "profile-hmm-viterbi"
    vecrec_bytes_per_cell = 4  # float H output
    report_gcups = False
    score_atol = 0.1  # vecrec/HMMER are float32; the numpy reference is float64
    boundary_convention = (
        "vecrec/HMMER: glocal profile Viterbi (UNIGLOCAL, special-state ribbon zeroed "
        "-> sequence fully consumed, model fully traversed); score = log-odds nats"
    )

    def __init__(self, rows: int = 128, cols: int = 128, seed: int = DEFAULT_SEED) -> None:
        # rows = observation length L, cols = model length M.
        self.rows = rows
        self.cols = cols
        self.seed = seed
        self.n_elements = rows * cols

    @classmethod
    def sweep_variants(cls) -> list[ProfileHmmViterbi]:
        return hmm_sweep_variants(cls)

    def make_inputs(self) -> dict[str, object]:
        return hmm_inputs(self.cols, self.rows, seed=self.seed)  # (M=cols, L=rows)

    def reference(self, **inputs: object) -> np.ndarray:
        return hmm_viterbi_reference(**inputs)

    def optimal_score(self, matrix: np.ndarray) -> float:
        return float(matrix[-1])  # corner H[L][M] = Viterbi optimum

    def kernel_expr(self) -> SignalExpr:
        return build_expr()

    def baselines(self) -> list[Baseline]:
        return [
            Baseline("numpy_loop", lambda **kw: hmm_viterbi_reference(**kw)),
        ]

    def cpp_baselines(self) -> list:
        from evals_common.cpp import HmmCppBaseline
        from evals_common.vecrec_hmm_cpp import vecrec_hmm_baselines

        return [
            *vecrec_hmm_baselines(self.kernel_expr()),
            HmmCppBaseline("hmmer_viterbi"),
        ]


if __name__ == "__main__":
    from evals_common.harness import run_cases

    raise SystemExit(run_cases([ProfileHmmViterbi()]))
