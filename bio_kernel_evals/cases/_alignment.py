"""Shared helpers for the sequence-alignment cases.

The vecrec kernels and the numpy reference operate on a precomputed per-cell
score matrix ``s[i, j]``; the external C++ libraries instead take two sequences
plus a substitution scoring scheme. To compare them on identical problems we
generate sequences ``x`` (length = rows) and ``y`` (length = cols) over a DNA
alphabet, then derive ``s[i, j] = subst[x_i, y_j]``. The case feeds ``s`` to
vecrec/reference and ``x, y`` + scoring to the C++ drivers.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np

ALPHABET = "ACGT"

# Square problem sizes for ``bio-evals --sweep`` (powers of two up to ~1k).
SWEEP_SIZES: tuple[int, ...] = (64, 128, 256, 512, 1024)

# Larger sweep for scaling studies (``bio-evals --sweep-sizes``).
LARGE_SWEEP_SIZES: tuple[int, ...] = (1024, 2048, 4096, 8192, 10000)

_CaseT = TypeVar("_CaseT")


def alignment_sweep_variants(
    case_cls: type[_CaseT],
    *,
    sizes: tuple[int, ...] = SWEEP_SIZES,
    seed: int = 0,
) -> list[_CaseT]:
    """Instantiate ``case_cls`` at each square size in ``sizes``."""
    return [case_cls(rows=n, cols=n, seed=seed) for n in sizes]


def banded_sweep_variants(
    case_cls: type[_CaseT],
    *,
    band_width: int,
    sizes: tuple[int, ...] = SWEEP_SIZES,
    seed: int = 0,
) -> list[_CaseT]:
    """Instantiate a banded alignment case at each square size in ``sizes``."""
    return [
        case_cls(rows=n, cols=n, band_width=band_width, seed=seed) for n in sizes
    ]
MATCH = 2
MISMATCH = -1


def substitution_matrix(match: int = MATCH, mismatch: int = MISMATCH) -> np.ndarray:
    """``subst[a, b] = match`` if ``a == b`` else ``mismatch`` over the alphabet."""
    k = len(ALPHABET)
    m = np.full((k, k), mismatch, dtype=np.int64)
    np.fill_diagonal(m, match)
    return m


def sequence_inputs(
    rows: int,
    cols: int,
    *,
    gap_open: int,
    gap_extend: int,
    match: int = MATCH,
    mismatch: int = MISMATCH,
    seed: int = 0,
) -> dict[str, object]:
    """Build a case input dict bridging sequences and the derived score matrix.

    ``gap_open`` / ``gap_extend`` are positive penalty magnitudes handed to the
    C++ drivers (the numpy/vecrec side bakes its own gap constant into the
    kernel). Returns ``s`` (int64 ``rows x cols``), the sequences ``x``/``y``,
    and the scoring parameters.
    """
    rng = np.random.default_rng(seed)
    subst = substitution_matrix(match, mismatch)
    xi = rng.integers(0, len(ALPHABET), rows)
    yj = rng.integers(0, len(ALPHABET), cols)
    s = subst[np.ix_(xi, yj)].astype(np.int32)
    x = "".join(ALPHABET[int(i)] for i in xi)
    y = "".join(ALPHABET[int(j)] for j in yj)
    return {
        "s": s,
        "x": x,
        "y": y,
        "match": match,
        "mismatch": mismatch,
        "gap_open": gap_open,
        "gap_extend": gap_extend,
    }


# Target identity for the WFA2-only "similar pair" case variants. y is a
# substitution-mutated copy of x (not an independent random draw).
DEFAULT_SIMILARITY = 0.65


def similar_sequences(
    rows: int,
    cols: int,
    *,
    identity: float = DEFAULT_SIMILARITY,
    seed: int = 0,
) -> tuple[str, str, float]:
    """Build a DNA pair with approximately ``identity`` fraction of matching bases.

    Square only (``rows == cols``). ``x`` is random DNA; ``y`` is a copy of ``x``
    with ``round((1 - identity) * n)`` positions flipped to a different base
    (substitution-only). Returns ``(x, y, actual_identity)``.
    """
    if rows != cols:
        raise ValueError(
            f"similar_sequences requires a square problem (got {rows}x{cols})"
        )
    n = rows
    rng = np.random.default_rng(seed)
    xi = rng.integers(0, len(ALPHABET), n)
    yj = xi.copy()
    n_mut = int(round(n * (1.0 - identity)))
    if n_mut > 0:
        mut_idx = rng.choice(n, size=n_mut, replace=False)
        for i in mut_idx:
            choices = [b for b in range(len(ALPHABET)) if b != int(xi[i])]
            yj[i] = choices[int(rng.integers(0, len(choices)))]
    x = "".join(ALPHABET[int(i)] for i in xi)
    y = "".join(ALPHABET[int(j)] for j in yj)
    actual = float(np.mean(xi == yj))
    return x, y, actual


def similar_sequence_inputs(
    rows: int,
    cols: int,
    *,
    gap_open: int,
    gap_extend: int,
    match: int = MATCH,
    mismatch: int = MISMATCH,
    identity: float = DEFAULT_SIMILARITY,
    seed: int = 0,
) -> dict[str, object]:
    """Like :func:`sequence_inputs` but with a ~``identity`` similar pair."""
    x, y, _actual = similar_sequences(rows, cols, identity=identity, seed=seed)
    subst = substitution_matrix(match, mismatch)
    xi = np.array([ALPHABET.index(c) for c in x])
    yj = np.array([ALPHABET.index(c) for c in y])
    s = subst[np.ix_(xi, yj)].astype(np.int32)
    return {
        "s": s,
        "x": x,
        "y": y,
        "match": match,
        "mismatch": mismatch,
        "gap_open": gap_open,
        "gap_extend": gap_extend,
    }


def similar_edit_cost_inputs(
    rows: int,
    cols: int,
    *,
    identity: float = DEFAULT_SIMILARITY,
    seed: int = 0,
) -> dict[str, object]:
    """Like :func:`edit_cost_inputs` but with a ~``identity`` similar pair."""
    x, y, _actual = similar_sequences(rows, cols, identity=identity, seed=seed)
    xi = np.array([ALPHABET.index(c) for c in x])
    yj = np.array([ALPHABET.index(c) for c in y])
    s = (xi[:, None] != yj[None, :]).astype(np.int32)
    return {
        "s": s,
        "x": x,
        "y": y,
        "match": 0,
        "mismatch": 1,
        "gap_open": 0,
        "gap_extend": 1,
    }


# Textbook dynamic-programming references.
#
# These compute the *standard* (Gotoh/Smith-Waterman/Needleman-Wunsch) optima so
# that the external C++ libraries can be validated by optimal-score equality.
# They use the same affine convention every C++ library here uses, namely a gap
# of length L costs ``gap_open + L * gap_extend`` (gap scores are negative).
#
# Note: vecrec's own kernels carry a slightly different boundary convention and
# are validated internally by vecrec's cross-schedule check; these references are
# the ground truth for the *baselines*, not a bit-for-bit model of the kernel.

_NEG = np.int64(np.iinfo(np.int64).min // 4)


def sw_reference(s: np.ndarray, gap: int) -> np.ndarray:
    """Smith-Waterman (local) DP with linear ``gap`` (negative). Returns H."""
    s = s.astype(np.int64)
    rows, cols = s.shape
    H = np.zeros((rows + 1, cols + 1), dtype=np.int64)
    for i in range(1, rows + 1):
        row = s[i - 1]
        for j in range(1, cols + 1):
            H[i, j] = max(
                0,
                int(H[i - 1, j - 1]) + int(row[j - 1]),
                int(H[i - 1, j]) + gap,
                int(H[i, j - 1]) + gap,
            )
    return H[1:, 1:].copy()


def nw_reference(s: np.ndarray, gap: int) -> np.ndarray:
    """Needleman-Wunsch (global) DP with linear ``gap`` (negative). Returns H."""
    s = s.astype(np.int64)
    rows, cols = s.shape
    H = np.zeros((rows + 1, cols + 1), dtype=np.int64)
    for i in range(1, rows + 1):
        H[i, 0] = i * gap
    for j in range(1, cols + 1):
        H[0, j] = j * gap
    for i in range(1, rows + 1):
        row = s[i - 1]
        for j in range(1, cols + 1):
            H[i, j] = max(
                int(H[i - 1, j - 1]) + int(row[j - 1]),
                int(H[i - 1, j]) + gap,
                int(H[i, j - 1]) + gap,
            )
    return H[1:, 1:].copy()


def affine_reference(s: np.ndarray, gap_open: int, gap_extend: int) -> np.ndarray:
    """Global affine (Gotoh) DP. ``gap_open``/``gap_extend`` are negative scores;
    a gap of length L costs ``gap_open + L * gap_extend``. Returns best-state H."""
    s = s.astype(np.int64)
    rows, cols = s.shape
    M = np.full((rows + 1, cols + 1), _NEG, dtype=np.int64)   # ends in (mis)match
    Dx = np.full((rows + 1, cols + 1), _NEG, dtype=np.int64)  # gap in y (consume x)
    Dy = np.full((rows + 1, cols + 1), _NEG, dtype=np.int64)  # gap in x (consume y)

    M[0, 0] = 0
    for i in range(1, rows + 1):
        Dx[i, 0] = gap_open + i * gap_extend
    for j in range(1, cols + 1):
        Dy[0, j] = gap_open + j * gap_extend

    for i in range(1, rows + 1):
        row = s[i - 1]
        for j in range(1, cols + 1):
            best_diag = max(int(M[i - 1, j - 1]), int(Dx[i - 1, j - 1]), int(Dy[i - 1, j - 1]))
            M[i, j] = best_diag + int(row[j - 1])
            Dx[i, j] = max(int(M[i - 1, j]) + gap_open + gap_extend, int(Dx[i - 1, j]) + gap_extend)
            Dy[i, j] = max(int(M[i, j - 1]) + gap_open + gap_extend, int(Dy[i, j - 1]) + gap_extend)

    H = np.maximum(np.maximum(M, Dx), Dy)
    return H[1:, 1:].copy()


def dual_affine_reference(
    s: np.ndarray, go: int, ge: int, go2: int, ge2: int
) -> np.ndarray:
    """Global 2-piece (dual) affine DP -- the reference for ksw_extd2_sse.

    All gap params are negative scores; a gap of length L costs
    ``max(go + L*ge, go2 + L*ge2)`` (the max of two affine pieces). Returns the
    best-state H; the global optimum is the corner ``H[-1, -1]``.
    """
    s = s.astype(np.int64)
    rows, cols = s.shape
    NEG = _NEG
    M   = np.full((rows + 1, cols + 1), NEG, dtype=np.int64)  # ends in (mis)match
    Dx  = np.full((rows + 1, cols + 1), NEG, dtype=np.int64)  # gap in y, piece 1
    Dy  = np.full((rows + 1, cols + 1), NEG, dtype=np.int64)  # gap in x, piece 1
    Dx2 = np.full((rows + 1, cols + 1), NEG, dtype=np.int64)  # gap in y, piece 2
    Dy2 = np.full((rows + 1, cols + 1), NEG, dtype=np.int64)  # gap in x, piece 2
    M[0, 0] = 0
    for i in range(1, rows + 1):
        Dx[i, 0]  = go  + i * ge
        Dx2[i, 0] = go2 + i * ge2
    for j in range(1, cols + 1):
        Dy[0, j]  = go  + j * ge
        Dy2[0, j] = go2 + j * ge2
    for i in range(1, rows + 1):
        row = s[i - 1]
        for j in range(1, cols + 1):
            best_diag = max(int(M[i - 1, j - 1]), int(Dx[i - 1, j - 1]),
                            int(Dy[i - 1, j - 1]), int(Dx2[i - 1, j - 1]),
                            int(Dy2[i - 1, j - 1]))
            M[i, j]   = best_diag + int(row[j - 1])
            Dx[i, j]  = max(int(M[i - 1, j]) + go  + ge,  int(Dx[i - 1, j])  + ge)
            Dy[i, j]  = max(int(M[i, j - 1]) + go  + ge,  int(Dy[i, j - 1])  + ge)
            Dx2[i, j] = max(int(M[i - 1, j]) + go2 + ge2, int(Dx2[i - 1, j]) + ge2)
            Dy2[i, j] = max(int(M[i, j - 1]) + go2 + ge2, int(Dy2[i, j - 1]) + ge2)
    H = np.maximum.reduce([M, Dx, Dy, Dx2, Dy2])
    return H[1:, 1:].copy()


def affine_sw_reference(s: np.ndarray, gap_open: int, gap_extend: int) -> np.ndarray:
    """Local affine (Smith-Waterman + Gotoh) DP -- the reference for ksw_ll_i16.

    ``gap_open``/``gap_extend`` are negative scores; a gap of length L costs
    ``gap_open + L * gap_extend``. Every cell may restart from 0 (local), and the
    two gap states open from the restarted H. Returns the restarted H matrix; the
    local optimum is ``H.max()``.
    """
    s = s.astype(np.int64)
    rows, cols = s.shape
    NEG = _NEG
    H = np.zeros((rows + 1, cols + 1), dtype=np.int64)         # restarted (>= 0)
    Dx = np.full((rows + 1, cols + 1), NEG, dtype=np.int64)    # gap in y (consume x)
    Dy = np.full((rows + 1, cols + 1), NEG, dtype=np.int64)    # gap in x (consume y)
    for i in range(1, rows + 1):
        row = s[i - 1]
        for j in range(1, cols + 1):
            Dx[i, j] = max(int(H[i - 1, j]) + gap_open + gap_extend, int(Dx[i - 1, j]) + gap_extend)
            Dy[i, j] = max(int(H[i, j - 1]) + gap_open + gap_extend, int(Dy[i, j - 1]) + gap_extend)
            m = int(H[i - 1, j - 1]) + int(row[j - 1])
            H[i, j] = max(0, m, int(Dx[i, j]), int(Dy[i, j]))
    return H[1:, 1:].copy()


def banded_nw_reference(s: np.ndarray, gap: int, band_width: int) -> np.ndarray:
    """Banded Needleman-Wunsch DP. Only cells within ``|i-j| <= band_width`` are
    computed; out-of-band cells are -inf.  This is the correct reference for any
    banded global aligner — both VecRec and SeqAn3 should agree with it regardless
    of whether bw captures the global optimum."""
    s = s.astype(np.int64)
    rows, cols = s.shape
    NEG = _NEG
    H = np.full((rows + 1, cols + 1), NEG, dtype=np.int64)
    H[0, 0] = 0
    for i in range(1, min(band_width + 1, rows + 1)):
        H[i, 0] = i * gap
    for j in range(1, min(band_width + 1, cols + 1)):
        H[0, j] = j * gap
    for i in range(1, rows + 1):
        row = s[i - 1]
        j_lo = max(1, i - band_width)
        j_hi = min(cols + 1, i + band_width + 1)
        for j in range(j_lo, j_hi):
            diag = int(H[i - 1, j - 1]) + int(row[j - 1]) if H[i - 1, j - 1] != NEG else NEG
            up   = int(H[i - 1, j])     + gap             if H[i - 1, j]     != NEG else NEG
            left = int(H[i, j - 1])     + gap             if H[i, j - 1]     != NEG else NEG
            H[i, j] = max(diag, up, left)
    return H[1:, 1:].copy()


def banded_sw_reference(s: np.ndarray, gap: int, band_width: int) -> np.ndarray:
    """Banded Smith-Waterman DP. Only cells within ``|i-j| <= band_width`` are
    computed; out-of-band cells use 0 (the local-alignment floor)."""
    s = s.astype(np.int64)
    rows, cols = s.shape
    H = np.zeros((rows + 1, cols + 1), dtype=np.int64)
    for i in range(1, rows + 1):
        row = s[i - 1]
        j_lo = max(1, i - band_width)
        j_hi = min(cols + 1, i + band_width + 1)
        for j in range(j_lo, j_hi):
            H[i, j] = max(
                0,
                int(H[i - 1, j - 1]) + int(row[j - 1]),
                int(H[i - 1, j]) + gap,
                int(H[i, j - 1]) + gap,
            )
    return H[1:, 1:].copy()


def banded_affine_reference(
    s: np.ndarray, gap_open: int, gap_extend: int, band_width: int
) -> np.ndarray:
    """Banded affine (Gotoh) DP. Only cells within ``|i-j| <= band_width`` are
    computed; out-of-band cells are -inf."""
    s = s.astype(np.int64)
    rows, cols = s.shape
    NEG = _NEG
    M  = np.full((rows + 1, cols + 1), NEG, dtype=np.int64)
    Dx = np.full((rows + 1, cols + 1), NEG, dtype=np.int64)
    Dy = np.full((rows + 1, cols + 1), NEG, dtype=np.int64)
    M[0, 0] = 0
    for i in range(1, min(band_width + 1, rows + 1)):
        Dx[i, 0] = gap_open + i * gap_extend
    for j in range(1, min(band_width + 1, cols + 1)):
        Dy[0, j] = gap_open + j * gap_extend
    for i in range(1, rows + 1):
        row = s[i - 1]
        j_lo = max(1, i - band_width)
        j_hi = min(cols + 1, i + band_width + 1)
        for j in range(j_lo, j_hi):
            def _safe_max(*vals):
                return max(v for v in vals if v != NEG) if any(v != NEG for v in vals) else NEG
            best_diag = _safe_max(int(M[i-1,j-1]), int(Dx[i-1,j-1]), int(Dy[i-1,j-1]))
            M[i, j]  = best_diag + int(row[j - 1]) if best_diag != NEG else NEG
            Dx[i, j] = _safe_max(
                int(M[i-1,j])  + gap_open + gap_extend if M[i-1,j]  != NEG else NEG,
                int(Dx[i-1,j]) + gap_extend             if Dx[i-1,j] != NEG else NEG,
            )
            Dy[i, j] = _safe_max(
                int(M[i,j-1])  + gap_open + gap_extend if M[i,j-1]  != NEG else NEG,
                int(Dy[i,j-1]) + gap_extend             if Dy[i,j-1] != NEG else NEG,
            )
    H = np.where((M == NEG) & (Dx == NEG) & (Dy == NEG), NEG,
                 np.maximum(np.maximum(M, Dx), Dy))
    return H[1:, 1:].copy()


def edit_cost_inputs(rows: int, cols: int, *, seed: int = 0) -> dict[str, object]:
    """Inputs for the unit-cost edit-distance case.

    ``s[i, j]`` is the substitution *cost* (0 if ``x_i == y_j`` else 1) for the
    vecrec TropMin kernel; ``x``/``y`` are the sequences for edlib. The scoring
    fields are placeholders (edlib uses fixed unit costs)."""
    rng = np.random.default_rng(seed)
    xi = rng.integers(0, len(ALPHABET), rows)
    yj = rng.integers(0, len(ALPHABET), cols)
    s = (xi[:, None] != yj[None, :]).astype(np.int32)  # 0 if equal else 1
    x = "".join(ALPHABET[int(i)] for i in xi)
    y = "".join(ALPHABET[int(j)] for j in yj)
    return {
        "s": s,
        "x": x,
        "y": y,
        "match": 0,
        "mismatch": 1,
        "gap_open": 0,
        "gap_extend": 1,
    }


def levenshtein_reference(x: str, y: str) -> np.ndarray:
    """Unit-cost Levenshtein DP. Returns D (without the 0-th row/col)."""
    n, m = len(x), len(y)
    D = np.zeros((n + 1, m + 1), dtype=np.int64)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        xi = x[i - 1]
        for j in range(1, m + 1):
            cost = 0 if xi == y[j - 1] else 1
            D[i, j] = min(
                int(D[i - 1, j - 1]) + cost,
                int(D[i - 1, j]) + 1,
                int(D[i, j - 1]) + 1,
            )
    return D[1:, 1:].copy()
