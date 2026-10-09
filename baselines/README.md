# Native C++ baseline drivers

Each external alignment library is benchmarked by a tiny C++ "driver" we compile
against the library's source cloned into `../third_party/<lib>/` (gitignored).
The drivers are invoked as subprocesses by the Python harness
(`evals_common/cpp.py`).

## The contract

A driver:

1. Reads an `input.json` path from `argv[1]` via `baselines::parse_input` (see
   [`common/driver_io.hpp`](common/driver_io.hpp)).
2. Runs the alignment under `baselines::Timer::median_seconds(warmup, iters, fn)`
   so only the alignment call is timed (no process startup or I/O).
3. Prints exactly one JSON line to stdout via `baselines::print_result`.

Input (written by the harness):

```json
{"algo":"sw","x":"ACGT...","y":"TTGA...","match":2,"mismatch":-1,
 "gap_open":1,"gap_extend":1,"warmup":3,"iterations":20}
```

`gap_open`/`gap_extend` are positive penalty magnitudes; each driver maps them to
its library's own gap convention (see the per-driver source comments).

Output (read by the harness):

```json
{"score": 123, "seconds": 0.000021, "gcups": 12.3, "cells": 65536}
```

`cells = len(x) * len(y)` is the DP-matrix size and the GCUPS denominator.

## Adding a library

1. Add it to `scripts/setup_baselines.sh` (clone into `third_party/<lib>` at a
   pinned commit).
2. Create `baselines/<lib>/<driver>.cpp` and `baselines/<lib>/CMakeLists.txt`
   that defines an executable linked against `baselines_common` and the library.
3. The top-level [`CMakeLists.txt`](CMakeLists.txt) auto-includes `<lib>/` once
   both the clone and the driver's `CMakeLists.txt` exist.
4. Reference the binary from a case via `CppBaseline` in
   `bio_kernel_evals/cases/`.

## parasail vectorization variants

The `parasail_align` driver dispatches on the ``algo`` field. Tags look like
``sw_striped_16``, ``nw_diag_32``, ``nw_stats``, etc.:

| Suffix | parasail family |
| --- | --- |
| `stats` | scalar reference |
| `striped_16` / `striped_32` | striped SIMD |
| `diag_16` / `diag_32` | antidiagonal SIMD |
| `scan_16` / `scan_32` | prefix-scan SIMD |

Cases use ``evals_common.parasail.parasail_baselines("sw")`` (or ``"nw"``) to
wire all variants. Bare ``sw`` / ``nw`` still map to ``striped_16`` for
backward compatibility.
