"""Bio-kernel evaluation suite.

Benchmarks vecrec's bioinformatics alignment kernels (Smith-Waterman,
Needleman-Wunsch, affine-gap, ...) against native C++ alignment libraries and a
numpy reference. Cases live under :mod:`bio_kernel_evals.cases` and are
auto-discovered by :func:`evals_common.harness.run_suite` via the ``bio-evals``
console script (see :mod:`bio_kernel_evals.run`).
"""
