"""Shared helpers for the profile-HMM Viterbi case.

Unlike the alignment cases (which generate sequences + a substitution matrix in
Python), the HMM model is owned by HMMER: the ``hmmer_viterbi`` driver's
``hmm_prepare`` mode samples a seeded profile HMM, configures it UNIGLOCAL, zeros
the special-state ribbon (so the Viterbi optimum is exactly the core M/I/D glocal
path), and dumps the exact configured scores it reads via ``TSC/MSC/ISC`` plus the
seeded observation. We consume that JSON so the numpy reference and the vecrec
kernel compute byte-for-byte the same DP as ``p7_GViterbi`` -- no re-derivation of
HMMER's ProfileConfig transforms, no quantization (everything is float32-range).

The recurrence (tropical max-plus), over sequence position ``i`` x model position
``k`` (HMMER ``generic_viterbi.c`` with the ribbon zeroed)::

    M[i][k] = EmitM(k,x_i) + max( M[i-1][k-1]+TMM(k-1), I[i-1][k-1]+TIM(k-1),
                                  D[i-1][k-1]+TDM(k-1), B[i-1]+TBM(k-1) )
    I[i][k] = EmitI(k,x_i) + max( M[i-1][k]+TMI(k), I[i-1][k]+TII(k) )   (k < M)
    D[i][k] = max( M[i][k-1]+TMD(k-1), D[i][k-1]+TDD(k-1) )

with ``B[0]=0`` (begin only at row 0; flanking forbidden), and the glocal optimum
``max(M[L][M], D[L][M])`` (sequence fully consumed, model fully traversed).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import TypeVar

import numpy as np

# Square model/sequence sizes for the sweep (model length M == sequence length L).
SWEEP_SIZES: tuple[int, ...] = (64, 128, 256, 512, 1024)

# Easel treats seed 0 as "choose an arbitrary seed" (nondeterministic across
# processes), which would desync the prepare/run models. Use a fixed nonzero seed.
DEFAULT_SEED = 1

# Transition sub-indices into the flat tsc array: tsc[k*8 + s].
MM, IM, DM, BM, MD, DD, MI, II = range(8)
_NEG = -1e30  # finite sentinel the driver emits for HMMER's -inf entries.

_CaseT = TypeVar("_CaseT")


def hmm_sweep_variants(
    case_cls: type[_CaseT],
    *,
    sizes: tuple[int, ...] = SWEEP_SIZES,
    seed: int = DEFAULT_SEED,
) -> list[_CaseT]:
    """Instantiate ``case_cls`` at each square size (rows=L, cols=M = n)."""
    return [case_cls(rows=n, cols=n, seed=seed) for n in sizes]


def _bin_dir() -> Path:
    override = os.environ.get("VECREC_EVALS_BIN")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "bin"


def hmmer_binary() -> Path:
    return _bin_dir() / "hmmer_viterbi"


def hmm_inputs(M: int, L: int, *, seed: int = DEFAULT_SEED) -> dict[str, object]:
    """Run the ``hmm_prepare`` driver and return the model scores + observation.

    Raises ``FileNotFoundError`` if the HMMER driver is not built (the case then
    SKIPs cleanly -- HMMER is required to even generate this case's inputs).
    """
    binary = hmmer_binary()
    if not binary.exists():
        raise FileNotFoundError(f"hmmer_viterbi driver not built: {binary}")
    payload = json.dumps({"algo": "hmm_prepare", "seed": seed, "M": M, "L": L})
    proc = subprocess.run(
        [str(binary), "/dev/stdin"], input=payload, capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"hmm_prepare failed: {proc.stderr.strip()}")
    d = json.loads(proc.stdout)
    return {
        "M": int(d["M"]),
        "L": int(d["L"]),
        "K": int(d["K"]),
        "seed": seed,
        "obs": np.asarray(d["obs"], dtype=np.int32),
        "tsc": np.asarray(d["tsc"], dtype=np.float64),
        "emitM": np.asarray(d["emitM"], dtype=np.float64),
        "emitI": np.asarray(d["emitI"], dtype=np.float64),
        "prepared_json": proc.stdout,  # raw, fed to the vecrec driver in Phase 3
    }


def hmm_viterbi_reference(
    *,
    M: int,
    L: int,
    K: int,
    obs: np.ndarray,
    tsc: np.ndarray,
    emitM: np.ndarray,
    emitI: np.ndarray,
    **_: object,
) -> np.ndarray:
    """Core glocal M/I/D Viterbi (float). Returns ``H = max(M,I,D)`` over the
    ``L x M`` cell grid (excluding the boundary), flattened; ``H[-1]`` is the
    optimum ``max(M[L][M], D[L][M])`` (``I[L][M]`` is always -inf).

    M and I depend only on row ``i-1`` (vectorized across k); D is a within-row
    recurrence (sequential scan), exactly as in the vecrec kernel.
    """
    TS = tsc.reshape(M + 1, 8)
    EM = emitM.reshape(K, M + 1)
    EI = emitI.reshape(K, M + 1)
    NEG = -np.inf

    Mt = np.full((L + 1, M + 1), NEG)
    It = np.full((L + 1, M + 1), NEG)
    Dt = np.full((L + 1, M + 1), NEG)
    B = NEG * np.ones(L + 1)
    B[0] = 0.0

    for i in range(1, L + 1):
        x = int(obs[i - 1])
        pM, pI, pD = Mt[i - 1], It[i - 1], Dt[i - 1]
        # M[i][k], k=1..M  <- diagonal from prev row at k-1, plus begin entry.
        diag = np.maximum.reduce([
            pM[0:M] + TS[0:M, MM],
            pI[0:M] + TS[0:M, IM],
            pD[0:M] + TS[0:M, DM],
            B[i - 1] + TS[0:M, BM],
        ])
        Mt[i, 1:M + 1] = diag + EM[x, 1:M + 1]
        # I[i][k], k=1..M-1  <- vertical from prev row at k.
        if M > 1:
            It[i, 1:M] = np.maximum(pM[1:M] + TS[1:M, MI], pI[1:M] + TS[1:M, II]) + EI[x, 1:M]
        # D[i][k], k=1..M  <- within-row recurrence (sequential).
        row_M, row_D = Mt[i], Dt[i]
        for k in range(1, M + 1):
            row_D[k] = max(row_M[k - 1] + TS[k - 1, MD], row_D[k - 1] + TS[k - 1, DD])

    H = np.maximum(np.maximum(Mt, It), Dt)
    return H[1:, 1:].reshape(-1).copy()
