# vecrec-small-comparison

Benchmarks VecRec-generated kernels against efficient hand-written C++ serial loops,
for the recurrence kernels listed in `benchmark-status.csv`.

For each benchmark:

1. **VecRec side** — `scripts/gen_kernels.py` drives VecRec to *find the best schedule*
   for the recurrence and emits a **self-contained** C++ kernel header to
   `generated/<kernel>.h` (compilable with only `clang++`, no Python at runtime).
2. **C++ side** — `benchmarks/<kernel>.cpp` times the VecRec kernel against an
   efficient serial loop, verifies they agree, and reports throughput + speedup,
   all through the centralized harness `harness/bench.hpp`.

## Layout

```
pyproject.toml          uv project; path-depends on the vendored ./vec-rec
harness/bench.hpp       centralized C++ harness: timing, RNG, verify, report, generic VecRec runner
specs/registry.py       benchmark name -> {kernel_name, build_expr()} (the recurrence specs)
scripts/gen_kernels.py  drive VecRec: best schedule -> generated/<kernel>.h (+ .meta.json)
scripts/run_all.py      generate + build + run + aggregate -> results.csv and a table
benchmarks/<kernel>.cpp one per benchmark: serial baseline + main()
Makefile                build/run individual benchmarks with clang++
generated/, build/      git-ignored (regenerated on demand)
vec-rec/                vendored VecRec checkout (git-ignored)
```

## Usage

```bash
uv sync                                              # set up the env (builds vecrec editable)

# Generate kernel header(s) — slow: runs VecRec's schedule search + sweep.
uv run python scripts/gen_kernels.py --list
uv run python scripts/gen_kernels.py second-order-low-pass
uv run python scripts/gen_kernels.py --all

# Build + run one benchmark (requires its generated header).
make second_order_low_pass

# Generate + build + run everything and aggregate results.
uv run python scripts/run_all.py
uv run python scripts/run_all.py --iters 200 --warmup 20
```

Benchmark source files and binaries are named after the kernel (underscored) so they
match the generated header. Optional CLI flags on each binary: `--rows --cols --iters
--warmup --seed`.

## Semantics notes

- **1D recurrences** (e.g. second-order low pass): VecRec treats the input as one
  contiguous stream (`reset_and_next_row` is a no-op for `Signal1D`), so the benchmark
  is a single long IIR (`--rows 1`) and the serial baseline is a single continuous loop.
- **2D recurrences** (SAT box filter, distance transform): the recurrence spans the
  whole grid; the serial baseline computes the full 2D recurrence.
- The harness rounds dimensions up to multiples of the kernel's `ele_per_iter` /
  `rows_per_iter` (a valid, slightly larger problem) so the run loop has no partial
  tiles and never reads/writes out of bounds.
- Correctness uses an absolute-or-relative tolerance: VecRec's vectorized (dilated)
  arithmetic reorders float operations, so it diverges slightly from the naive loop.

## VecRec tweaks (in the vendored `vec-rec/`, additive)

1. `src/vecrec/codegen/generate_kernel.py` — `find_best_schedule_and_emit_header(expr,
   header_path, kernel_name, baseline_kernel_name=..., ...)`: enumerate candidate schedules,
   time them, sweep codegen params on the best, and emit a standalone header containing
   **two** structs — `<kernel_name>` (best schedule) and `<kernel_name>_baseline` (the
   reference/un-optimized schedule `schedules[0]` with its own swept epi/rpi).
2. `src/vecrec/codegen/codegen.py` — each generated `struct` now also exposes
   `static constexpr int ele_per_iter`, `static constexpr int rows_per_iter`, and
   `using vector_type`, so the C++ harness can drive any kernel generically without a
   per-kernel calling template.

## Results

Every benchmark from `benchmark-status.csv` that is expressible in VecRec is implemented
(seam carving is excluded — its centered `min` reads a future column of the previous row, a
negative column lag that `TIKernel2D` cannot represent). Each compares **three**
implementations (`clang++ -O3 -ffast-math -march=native`, AVX-512 machine; 1D = a 2^22-sample
stream, 2D = a 2048×2048 grid — both 4,194,304 elements / 16 MiB per array; representative
numbers — timings vary with machine load):

- **serial** — efficient C++ serial loop.
- **VecRec baseline** — VecRec's *un-optimized* reference schedule (`schedules[0]`: no
  dilation/factorization) with its `ele_per_iter`/`rows_per_iter` swept. Same kernel a
  programmer would get from VecRec's codegen without the optimizer; a stronger baseline
  than the naive loop because it still gets register tiling / unrolling.
- **VecRec optimized** — the best schedule found by the search, with its best epi/rpi.

`opt/ser` is the headline speedup; `opt/base` isolates the gain attributable to VecRec's
*optimization* (dilation/factorization) rather than codegen alone.

| benchmark | type | opt/serial | opt/baseline | baseline/serial | status |
|---|---|---|---|---|---|
| exponential-smoothing | 1D arith | ~7.8× | ~6.4× | 1.22× | PASS |
| double-exp-smoothing (Holt's) | 1D arith (2×2 mutual) | ~5.2× | ~6.1× | 0.86× | PASS |
| box-filter-sat | 2D arith | ~6.2× | ~6.0× | 1.03× | PASS |
| repeated-box-filter | 2D arith (3-pass SAT) | ~11.7× | ~9.4× | 1.25× | PASS (10% tol) |
| first-order-high-pass | 1D arith | ~10.6× | ~7.5× | 1.41× | PASS |
| recursive-gaussian | 1D arith (3rd order) | ~9.8× | ~7.8× | 1.25× | PASS |
| second-order-high-pass | 1D arith | ~9.9× | ~5.8× | 1.71× | PASS |
| second-order-low-pass | 1D arith | ~6.3× | ~5.2× | 1.21× | PASS |
| distance-transform | 2D tropical (min-plus) | ~4.2× | ~5.1× | 0.83× | PASS (exact) |
| nonuniform-recursive-filter | 1D arith (time-varying) | ~3.0× | ~3.0× | 0.99× | PASS |
| welford-mean | 1D arith (time-varying, 1st order) | ~4.0× | ~3.9× | 1.03× | PASS |
| continued-fraction | 1D arith (time-varying, **float64**) | ~2.0× | ~1.8× | 1.13× | PASS |

With `-ffast-math` (FMA contraction enabled) and the codegen fix that stops the scalar
baseline from materializing a wide vector load and immediately splitting it into lanes, the
VecRec **baseline** beats (or ties) the serial loop on most kernels (`baseline/serial ≳ 1`) —
i.e. codegen's register tiling/unrolling alone, before any optimization, already wins. The
exceptions (`< 1`) are double-exp-smoothing (a coupled 2-variable mutual recurrence) and
distance-transform (where `-ffast-math` also lets the compiler vectorize the serial min-plus
loop); the time-varying kernels sit near `1.0` (their per-sample coefficient loads leave little
for codegen alone to win). Either way `opt/baseline` isolates a further gain from VecRec's
optimization — up to ~5–9× on the time-invariant filters, ~1.8–3× on the time-varying ones.
Correctness compares each VecRec kernel against the serial loop with an absolute-or-relative
tolerance (both built with `-ffast-math`); the baseline is usually numerically closer to serial
than the optimized kernel. repeated-box-filter uses a 10% bound — cascaded float32 SAT loses
precision with grid width (optimized error ~5.4e-2 at 2048², baseline ~6e-5). See `results.csv`
for exact errors.

### Time-varying (non-uniform) recurrences

The last three benchmarks have coefficients that vary per sample, encoded with VecRec's
`TVKernel` (each tap is an input *signal* rather than a constant):

- **nonuniform-recursive-filter** — `f(x)=a(x)·f(x-1)+b(x)·f(x-2)+g(x)`, a genuine time-varying
  2nd-order IIR (vec-rec's `second_order_tv`). Coefficients are drawn from `[-0.4, 0.4]` so
  `|a|+|b|<1` and the filter stays BIBO-stable over the stream.
- **welford-mean** — Welford's running mean `M(x)=M(x-1)+(a(x)-M(x-1))/x`. Rewritten as a
  first-order TV recurrence `M(x)=(1-w(x))·M(x-1)+w(x)·a(x)`. The weight `w(x)=1/x` is a pure
  function of position, so rather than materialize it, it is computed **on the fly** by a
  `ComputedSignal2D` — a small C++ struct that tracks the stream position and emits a
  fully-vectorized reciprocal (one `vdivps` per step, no memory traffic for `w`). Both the
  VecRec kernel and the serial baseline compute `1/x` on the fly, so the comparison is
  division-bound: VecRec's vectorized reciprocal beats the scalar serial one ~4×.
- **continued-fraction** — the Wallis–Euler recurrence `h(x)=a(x)·h(x-1)+b(x)·h(x-2)` with
  `f(x)=h(x)/h(x-1)`. `h` grows/decays geometrically, so a single long stream overflows within
  ~100 steps; this is the benchmark that motivated **adding `float64` support to vec-rec**. Even
  in double a long run overflows, so we run many *independent* 512-step continued fractions —
  **one CF per harness row** (`row_size=512`). The recurrence state resets at each row boundary
  (VecRec's recurrence zeroes its history in `reset_and_next_row`; only *input readers* stream
  continuously), so every row is a fresh CF; a seed input injects `h=1` at each row's first step.
  It is a real lag-1 recurrence, so VecRec dilates it to vectorize within each 512-step segment
  (~1.8× over the un-dilated baseline). Correctness excludes each row's first step, where
  `f=h(0)/h(-1)` divides by the zeroed post-reset history.

Regenerate everything with `uv run python scripts/run_all.py` (use `--regen` to redo the
schedule search).
