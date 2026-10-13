# bio_kernel_evals/

The isolated kernel benchmark suite. Compares VecRec-generated alignment kernels
against native C++ library baselines on correctness and throughput (GCUPS).

## Running

```bash
# One case (128×128 by default)
uv run python -m bio_kernel_evals.cases.needleman_wunsch
uv run python -m bio_kernel_evals.cases.affine_gap
uv run python -m bio_kernel_evals.cases.banded_needleman_wunsch

# Full suite (all cases, default sizes)
uv run python bio_kernel_evals/run.py

# Sweep over 64, 128, 256, 512, 1024
uv run python bio_kernel_evals/run.py --sweep

# Specific size for one case
uv run python -c "
from evals_common.harness import run_cases
from bio_kernel_evals.cases.affine_gap import AffineGap
run_cases([AffineGap(rows=512, cols=512)])
"
```

## What "bio-evals" means

The `bio-evals` console script (installed by `uv sync`) is an alias for
`uv run python bio_kernel_evals/run.py`. Both run the full suite.

## Output format

```
=== needleman-wunsch (128x128, seed=0) ===
  reference score: 80
  correctness:
    impl                    result   details
    vecrec_nw_tb:k5          PASS   score=80
    seqan3_nw                PASS   score=80
    ...
  throughput (wall time, lower is better):
    impl                    mode         median    vs_best   correct
    vecrec_nw_tb:k5         traceback    <time>    <ratio>   PASS
    parasail_nw_scan_16     traceback    <time>    <ratio>   PASS
    ...
  fastest vecrec schedule: k5 (<time>)
```

`vs_best` is relative to the fastest **correct** implementation. VecRec
correctness failures are reported but do not cause the suite to fail (VecRec
is the subject under test).

## Prerequisites

Build the C++ baseline libraries first:
```bash
bash scripts/build_baselines.sh
```

Cases whose binary is not found in `bin/` are reported as `SKIP`.

## Directory layout

```
bio_kernel_evals/
  run.py          — CLI entry point (bio-evals script)
  __init__.py
  cases/
    README.md     — case inventory, parameter guide, how to write a new case
    *.py          — one EvalCase subclass per algorithm
    _*.py         — private helpers (reference DPs, HMM kernel, etc.)
```

See `cases/README.md` for the full case inventory.
