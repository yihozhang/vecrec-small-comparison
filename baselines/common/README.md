# baselines/common/

Shared C++ headers included by every baseline driver. All are header-only.

## `driver_io.hpp` — JSON I/O + timing contract

Every driver must:
1. Call `baselines::parse_input(argv[1])` to read the input JSON
2. Time ONLY the alignment: `baselines::Timer::median_seconds(warmup, iters, fn)`
3. Print result: `baselines::print_result(score, seconds, cells)`

**Input JSON** (written by `evals_common/cpp.py`):
```json
{
  "algo": "nw",
  "x": "ACGT...", "y": "TTGA...",
  "match": 2, "mismatch": -1,
  "gap_open": 1, "gap_extend": 1,
  "warmup": 3, "iterations": 20
}
```

Gap values are **positive penalty magnitudes**. Each driver maps them to
its library's own sign convention (see per-driver source comments).

**Output JSON** (parsed by `evals_common/cpp.py`):
```json
{"score": 123, "seconds": 0.000021, "gcups": 12.3, "cells": 65536}
```

`cells = len(x) * len(y)` is the GCUPS denominator.

**`baselines::Input` struct fields:**
```cpp
struct Input {
    std::string algo;   // "nw", "sw", "affine", "nw_banded", ...
    std::string x, y;   // sequences
    double match, mismatch, gap_open, gap_extend;
    double gap_open2, gap_extend2;  // for dual-affine; 0 if not provided
    int warmup, iterations;
};
```

## `traceback.hpp` — O(n+m) traceback functions

Used by VecRec C++ drivers (not library baselines). Provides:

```cpp
// Full-matrix NW traceback
baselines::nw_traceback_full_inline(output, rows, cols, phys_w,
    x, y, gap, match, mismatch, q_out, r_out, cigar_out);

// Full-matrix affine traceback
baselines::affine_traceback_full_inline(output, rows, cols, phys_w,
    x, y, go, ge, match, mismatch, q_out, r_out, cigar_out);

// Banded NW traceback
baselines::nw_traceback_banded_inline(band_buf, col_starts, n_col,
    rows, cols, x, y, gap, match, mismatch, sentinel, q_out, r_out, cigar_out);

// Local SW traceback  
baselines::sw_traceback_full_inline(output, rows, cols, phys_w,
    x, y, gap, match, mismatch, q_out, r_out, cigar_out, &score_out);
```

All functions take the full DP output matrix and walk back from the optimum.

## `hmm_io.hpp` — HMM input/output

Used only by the HMMER baseline driver. Reads HMM profile parameters and
emission/transition matrices from the input JSON.
