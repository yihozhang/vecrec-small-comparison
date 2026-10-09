"""Helpers for wiring parasail vectorization variants as C++ baselines."""

from __future__ import annotations

from evals_common.cpp import CppBaseline

# (variant suffix, short description for docs, check_correctness)
#
# check_correctness=False is used for cell widths that can silently saturate
# at the sweep sizes this harness runs (MATCH=2 means scores can exceed the
# int8 range ~127 once N > ~64) -- those variants still time, but never gate
# the suite. ``sat`` auto-widens on overflow so it's correctness-safe despite
# starting from 8-bit cells.
PARASAIL_VARIANTS: tuple[tuple[str, str, bool], ...] = (
    ("stats", "scalar stats (correctness reference)", True),
    ("striped_16", "striped SIMD, 16-bit cells", True),
    ("diag_16", "antidiagonal SIMD, 16-bit cells", True),
    ("scan_16", "prefix-scan SIMD, 16-bit cells", True),
    ("striped_32", "striped SIMD, 32-bit cells", True),
    ("diag_32", "antidiagonal SIMD, 32-bit cells", True),
    ("scan_32", "prefix-scan SIMD, 32-bit cells", True),
    ("striped_64", "striped SIMD, 64-bit cells", True),
    ("diag_64", "antidiagonal SIMD, 64-bit cells", True),
    ("scan_64", "prefix-scan SIMD, 64-bit cells", True),
    ("striped_sat", "striped SIMD, 8-bit cells auto-widened on overflow", True),
    ("diag_sat", "antidiagonal SIMD, 8-bit cells auto-widened on overflow", True),
    ("scan_sat", "prefix-scan SIMD, 8-bit cells auto-widened on overflow", True),
    ("striped_8", "striped SIMD, 8-bit cells (timing only -- saturates at N>~64)", False),
    ("diag_8", "antidiagonal SIMD, 8-bit cells (timing only -- saturates at N>~64)", False),
    ("scan_8", "prefix-scan SIMD, 8-bit cells (timing only -- saturates at N>~64)", False),
)


def parasail_baselines(
    align: str,
    *,
    prefix: str | None = None,
    variants: tuple[str, ...] | None = None,
) -> list[CppBaseline]:
    """Return :class:`CppBaseline` entries for parasail implementation variants.

    Args:
        align: ``"sw"`` or ``"nw"`` (Smith-Waterman vs Needleman-Wunsch).
        prefix: Optional name prefix (e.g. ``"affine"`` → ``parasail_affine_striped_16``).
        variants: Subset of variant suffixes; defaults to all in :data:`PARASAIL_VARIANTS`.
    """
    if align not in ("sw", "nw"):
        raise ValueError(f"align must be 'sw' or 'nw', got {align!r}")

    correctness_by_variant = {v: ok for v, _, ok in PARASAIL_VARIANTS}
    allowed = set(correctness_by_variant)
    chosen = variants or tuple(v for v, _, _ in PARASAIL_VARIANTS)
    unknown = set(chosen) - allowed
    if unknown:
        raise ValueError(f"unknown parasail variants: {sorted(unknown)}")

    label = prefix or align
    out: list[CppBaseline] = []
    for variant in chosen:
        out.append(
            CppBaseline(
                name=f"parasail_{label}_{variant}",
                binary="parasail_align",
                algo=f"{align}_{variant}",
                check_correctness=correctness_by_variant[variant],
            )
        )
    return out
