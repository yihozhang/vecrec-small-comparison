# evals_common/

Suite-agnostic Python harness and VecRec compilation infrastructure. These
modules are shared across all eval suites in this repo.

## Module overview

### `harness.py` — core benchmark framework

The main harness. Key classes:

```python
class EvalCase(ABC):
    """One algorithm to benchmark. Subclass and implement 4 methods."""
    name: str                      # shown in output tables
    def inputs(self) -> dict       # {"x": str, "y": str, ...}
    def vecrec_expr(self)          # returns a VecRec SignalExpr
    def optimal_score(self, inputs) -> float  # reference score for correctness
    def baselines(self) -> list[Baseline]     # C++ and Python competitors

class Baseline:
    """A Python callable competitor. Timed in-process with warmup."""
    def __init__(self, name, fn)

def run_cases(cases: list[EvalCase]) -> None:
    """Run one or more cases and print the unified result table."""
```

Run a single case from the command line:
```bash
uv run python -m bio_kernel_evals.cases.needleman_wunsch
```

### `vecrec_cpp.py` — VecRec kernel compilation + C++ driver

Compiles a VecRec `SignalExpr` into a self-timed native C++ driver that:
1. Generates the kernel header via VecRec codegen
2. Wraps it in a C++ driver with the same `driver_io.hpp` contract as the
   library baselines (reads `input.json`, prints one JSON line)
3. Times `alloc + kernel + traceback` as a single region under
   `baselines::Timer`

Key function:
```python
def vecrec_traceback_baselines(expr, kind: str) -> list[VecrecTracebackBaseline]:
    """One baseline per candidate schedule (k0, k1, ...).
    kind: "nw", "sw", "affine", or "levenshtein" — selects the traceback."""
```

Also contains `_band_col_bounds(band_width, rpi, epi)` which emits the
`col_start`/`col_end` C++ expressions for banded drivers.

### `cpp.py` — native C++ baseline runner

Runs a pre-compiled binary from `bin/`, parses its JSON output:
```python
class CppBaseline(Baseline):
    def __init__(self, name, binary_path, algo, extra_fields=None)
```

The binary receives an `input.json` via `argv[1]` and prints one JSON line.

### `metrics.py` — throughput + correctness metrics

Computes GCUPS, speedup vs best, p10/p50/p90 timing stats, and correctness
labels (`PASS`/`FAIL`/`SKIP`).

### `parasail.py` — parasail variant factory

```python
def parasail_baselines(algo: str) -> list[CppBaseline]:
    """Returns all parasail striped/diag/scan ×2 variants for 'nw' or 'sw'."""
```

### `vecrec_hmm_cpp.py` — HMM-specific VecRec driver

Like `vecrec_cpp.py` but for profile HMM Viterbi kernels.

## How VecRec timing works

Every VecRec schedule is compiled into a standalone C++ binary under a temp dir.
The binary:
1. Reads `input.json` (sequences, scoring params, warmup/iters)
2. Runs `Timer::median_seconds(warmup, iters, [&]{ kernel(); traceback(); })`
3. Prints `{"score": N, "seconds": T, "gcups": G, "cells": C}`

This is identical to how the library baselines work, so comparisons are
apples-to-apples: every number includes allocation + DP + traceback.

## Adding a Python baseline

```python
from evals_common.harness import Baseline
import numpy as np

def my_numpy_nw(x, y, match, mismatch, gap_extend, **_):
    # ... textbook DP ...
    return score

# In your EvalCase.baselines():
def baselines(self):
    return [Baseline("numpy_nw", my_numpy_nw)]
```

## Adding a C++ baseline

See `baselines/README.md` for the driver contract. Once the binary is in `bin/`:

```python
from evals_common.cpp import CppBaseline

def baselines(self):
    return [CppBaseline("mylib_nw", "bin/mylib_nw", "nw")]
```
