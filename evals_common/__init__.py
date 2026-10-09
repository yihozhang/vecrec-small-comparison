"""Shared, suite-agnostic evaluation harness.

This package holds the pieces every evaluation suite reuses:

* :mod:`evals_common.harness` -- the ``EvalCase`` / ``Baseline`` abstractions and
  the runner that executes vecrec kernels plus external baselines.
* :mod:`evals_common.cpp` -- ``CppBaseline``, the bridge that runs a compiled
  native driver (under ``bin/``) and parses its JSON result.

Individual suites (``bio_kernel_evals``, ``leetcode_evals``, ...) provide only
their own ``cases`` package and a ``run`` entry point that calls
:func:`evals_common.harness.run_suite`.
"""
