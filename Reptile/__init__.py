"""Reptile — REPTILE-style tiling of VecRec alignment recurrences.

This package treats each VecRec inner-loop kernel as a *black box* (an external
function in REPTILE terms) and constructs tile loops around it, following the
tiling philosophy of REPTILE (Tariq, Sundram, Kjolstad, OOPSLA 2025). It targets
the two dense-DP recurrences from ``bio_kernel_evals``: Needleman-Wunsch and the
affine-gap (Gotoh) alignment.

See ``README.md`` for the mapping from the paper's constructs to this repo and
for the experimental findings.
"""
