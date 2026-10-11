# VecRec — Artifact Evaluation

This repository reproduces the evaluation of the VecRec paper. VecRec itself is
the Python library + C++ template header pinned as the git submodule
[`vec-rec/`](vec-rec); everything else here is benchmarking harness.

| Paper table | Group name | What is measured |
|---|---|---|
| Table 1 | `wild` | 12 "recurrences in the wild" (signal processing / numerics): VecRec kernel vs a hand-written serial C++ loop vs VecRec with vectorization disabled |
| Table 2 | `leetcode` | 12 LeetCode dynamic-programming recurrences, same three implementations |
| Table 3 | `bio` | Edit distance / Needleman–Wunsch / Smith–Waterman / affine gap (2048×2048): VecRec vs parasail, ksw2, seqan3, edlib, WFA2 |
| Table 4 | `e2e` | minimap2 end-to-end on three real ONT datasets, stock ksw2 vs the VecRec drop-in kernel |
| Table 5 | `reptile` | Smith–Waterman: REPTILE-style scalar tiling vs VecRec |

Two commands drive everything:

```bash
uv run bench exec <group>     # run the experiments behind one table  (results/<group>/<host>.csv)
uv run fig <group>            # render that table as LaTeX + Markdown  (figures/<group>.{tex,md})
```

The paper reports every table on three machines (Intel Skylake i9-9960X, AMD
Zen 5 Threadripper 9970X, Apple M2). The scripts run on **one** machine and
produce a one-machine table; `fig` lays several machines' result files side by
side when given more than one, which is how the paper's tables were assembled.
The paper's own numbers are transcribed under [`results/paper/`](results/paper)
so you can render them next to your reproduction (see "Comparing with the paper").

## 1. Requirements

* Linux x86-64 or macOS/arm64. A machine with AVX2/AVX-512 (x86) or NEON (Apple
  Silicon). All experiments are single-threaded.
* **clang ≥ 17** as `clang++` on `PATH` (VecRec kernels use clang's
  `ext_vector_type` and elementwise builtins; GCC does not work. seqan3 also
  needs clang ≥ 17). On macOS use Homebrew LLVM (`brew install llvm`), not Apple's.
* [`uv`](https://docs.astral.sh/uv/) (Python ≥ 3.10 is fetched by uv if needed).
* `git`, `make`, `cmake ≥ 3.20`, `curl`, `gunzip`, and zlib development headers
  (`zlib1g-dev` on Debian/Ubuntu; present by default on macOS) — needed by the
  third-party aligners in Tables 3–5.
* Network access the first time Tables 3–5 are set up: the library baselines are
  cloned from GitHub at pinned versions and the Table 4 datasets are downloaded
  from NCBI / GitHub (≈ 60 MB).
* Disk: ≈ 2 GB (third-party clones + builds).

No Python packages beyond `numpy` and the vendored `vecrec` are used.

## 2. Setup

```bash
git clone --recursive https://github.com/yihozhang/vecrec-small-comparison
cd vecrec-small-comparison
uv sync                          # creates .venv with vecrec (editable) + numpy
uv run bench list                # sanity check: prints groups and the host tag
```

If you cloned without `--recursive`, run `git submodule update --init` first.
`vec-rec/` must be at the commit this repository pins (`git ls-tree HEAD vec-rec`);
`bench` refuses to run otherwise, because VecRec's default branch does not yet
contain the banded-kernel support the Table 3-5 harnesses use.

Tables 1–2 need nothing else. Tables 3–5 need the C++ baselines and datasets;
`bench exec` fetches/builds them on demand, or do it explicitly:

```bash
uv run bench setup bio           # clone + build parasail, ksw2, seqan3, edlib, WFA2  (~10 min)
uv run bench setup e2e           # download the 3 ONT datasets + build minimap2 (stock & VecRec)
```

## 3. Running

```bash
uv run bench exec wild           # Table 1
uv run bench exec leetcode       # Table 2
uv run bench exec bio            # Table 3
uv run bench exec e2e            # Table 4
uv run bench exec reptile        # Table 5 (meaningful on Apple Silicon only, see §6)
uv run bench exec all            # everything
```

Results land in `results/<group>/<host>.csv` (`.json` for `reptile`), where
`<host>` is `--host NAME`, else `$AE_HOST`, else the hostname. Re-running a
group overwrites that file; partial results are flushed after every benchmark.

Useful flags:

* `--quick` — smoke-test settings (tiny sizes, few iterations). Finishes in
  minutes and exercises every code path, but the numbers are **not** paper-grade.
* `--only NAME` — one benchmark (`wild`/`leetcode`: registry name such as
  `second-order-low-pass`; `bio`: case name such as `affine-gap`;
  `e2e`: dataset label such as `pa-strains`). Repeatable.
* `--regen` (`wild`/`leetcode`) — redo the VecRec schedule search instead of
  reusing `generated/<kernel>.h` from a previous run. The search is the slow
  part (it compiles and times every candidate schedule); the headers are cached
  per machine because the best schedule is machine-specific.
* `--reps N` (`e2e`, `reptile`), `--size N` (`bio`), `--iters/--warmup` (`wild`/`leetcode`).

To start over, `uv run bench clean <group>` deletes what that group rebuilds on
demand (kernel headers and binaries for `wild`/`leetcode`, which forces a new
schedule search; library drivers for `bio`; the minimap2 binaries and generated
kernel for `e2e`; `Reptile/build/`). `--deep` also removes `third_party/` and the
downloaded datasets. Result files under `results/` are never touched.

Then render:

```bash
uv run fig wild                  # prints a Markdown table, writes figures/wild.{tex,md}
uv run fig all
```

`fig` uses every file in `results/<group>/` as one column group (labelled by
file stem). To choose files and labels explicitly:

```bash
uv run fig bio --source "Zen 5=results/bio/zen5.csv" --source "Apple M2=results/bio/m2.csv"
```

### Comparing with the paper

```bash
uv run fig wild --source "Skylake (paper)=results/paper/wild-skylake.csv" \
                --source "Zen 5 (paper)=results/paper/wild-zen5.csv" \
                --source "$(hostname)=results/wild/$(hostname).csv"
```

`results/paper/` holds the paper's tables re-encoded in the result-file schema
(`wild-*`, `leetcode-*`, `bio-*` for the three machines, `e2e-paper.csv`,
`reptile-m2.json`); rendering them alone reproduces the paper's tables verbatim.

## 4. What each group does, and what to expect

### Tables 1–2 (`wild`, `leetcode`)

For each recurrence in [`specs/registry.py`](specs/registry.py):

1. VecRec enumerates candidate schedules for the recurrence, times them, sweeps
   the codegen parameters (`ele_per_iter`, `rows_per_iter`) on the best ones and
   writes a self-contained header `generated/<kernel>.h` with two structs: the
   best schedule and the *baseline* schedule (VecRec codegen with the
   vectorizing transformations disabled). This is `Vec` and `Base` in the table.
2. `benchmarks/<kernel>.cpp` (the hand-written serial loop = `Serial`, plus the
   generic driver in [`harness/bench.hpp`](harness/bench.hpp)) is compiled with
   `clang++ -O3 -ffast-math -march=native`.
3. The binary times all three on the same input (2^22 elements: a 2^22-sample
   stream for 1-D recurrences, a 2048×2048 grid for 2-D ones), checks the two
   VecRec kernels against the serial loop, and prints a `RESULT,...` line.

Speedup = `Vec` over the faster of `Serial` and `Base`. Expect the same
ordering and magnitudes as the paper's column for the closest architecture;
absolute throughputs depend on the machine, and the schedule search itself is
timing-based, so the chosen schedule (and hence `Vec`) can vary by a few
percent between runs.

The paper's row names are hard-coded in [`ae/layout.py`](ae/layout.py); e.g. the
registry's `second-order-low-pass` is printed as "Second-Order Band-Pass Filter"
as in the paper.

### Table 3 (`bio`)

[`bio_kernel_evals/`](bio_kernel_evals) + [`evals_common/`](evals_common): for
each of the four recurrences the harness builds every candidate VecRec schedule
at int8 / int16 / int32, runs every library driver under `bin/` (sources in
[`baselines/`](baselines), each self-timing one complete alignment *including
traceback* and reporting its score), and validates every implementation's
optimal score against a textbook DP in Python. The table takes, per benchmark,
the fastest *correct* VecRec schedule of any element width (int8/int16/int32)
and the baselines the paper lists; the speedup is VecRec over the fastest correct
baseline. Note the paper's caption says "int16 and int32 variants", but the
script that produced its numbers took the fastest VecRec schedule of any width,
and for affine gap that is the int8 kernel (whose score check passes at
2048x2048); `fig bio --vecrec-widths i16,i32` applies the caption's rule instead,
which roughly halves the affine-gap VecRec throughput. The two extra
cases `*-sim65` / `*-sim85` run WFA2 alone on 65% / 85%-identity sequence pairs
(WFA2's runtime depends on similarity, see the paper's footnote); the 85% row
reports WFA2's speedup *over VecRec*. Edit distance is expected to lose to edlib.

For NW, SW and edit distance the int8 schedules fail the score check at
2048×2048 (scores exceed the int8 range) and are therefore never selected.

### Table 4 (`e2e`)

[`end_to_end_baselines/minimap2/`](end_to_end_baselines/minimap2): a vendored
minimap2 v2.28 with small instrumentation hooks, built twice — stock, and with
`ksw2_extz2_sse`/`ksw2_extd2_sse` replaced by the VecRec affine kernel
(`generated/ksw2_vecrec_ext.cpp`, generated by `scripts/generate_kernel.py`).
Each dataset is aligned single-threaded (`-t 1`) with `-x map-ont` and
banding / approximate-max / z-drop disabled (`-r 100000,100000 -z 1000000,1000000`,
the scenario `ablation-noapprox-*` in `scenarios.py`); the table reports the
median wall time of 5 runs and checks both binaries produce the same number of
alignments.

Datasets (downloaded into `end_to_end_baselines/data/realbench/`, gitignored):
`ecoli-ultralong` = 371 real ultra-long ONT *E. coli* reads vs K-12 MG1655;
`ecoli-strains` = *E. coli* O157:H7 chopped into 100 × 50 kbp contigs vs K-12;
`pa-strains` = *P. aeruginosa* PA14 contigs vs PAO1.

### Table 5 (`reptile`)

[`Reptile/`](Reptile): a REPTILE-style tiled scalar loop nest for
Smith–Waterman (untiled / row / column / 2-D block tiling, tile sizes 4–512) and
the VecRec kernel, every configuration validated against the reference score.
The table reports, per matrix size, the untiled scalar loops, the best scalar
tiling, and VecRec untiled. The harness also tiles *around* the VecRec kernel
(the paper's "composing VecRec with Reptile" remark); those rows are in the JSON.

## 5. Layout

```
ae/                      bench / fig entry points, paper row layouts, table renderers
specs/registry.py        Table 1-2 recurrence definitions (VecRec expressions)
benchmarks/*.cpp         Table 1-2 serial baselines + main()      harness/bench.hpp  driver
Makefile                 builds benchmarks/<kernel>.cpp against generated/<kernel>.h
bio_kernel_evals/        Table 3 cases (one file per recurrence)   evals_common/  harness
baselines/               C++ drivers for parasail/ksw2/seqan3/edlib/WFA2 (CMake)
scripts/setup_baselines.sh, build_baselines.sh     clone (pinned) + build -> bin/
end_to_end_baselines/    Table 4: vendored minimap2, kernel generator, scenarios, fetch scripts
Reptile/                 Table 5: tiled-driver generator + sweep
results/<group>/         raw results per machine; results/paper/ = the paper's numbers
figures/                 output of `fig`
vec-rec/                 VecRec (git submodule, pinned)
generated/ build/ bin/ third_party/   gitignored build products
```

## 6. Known deviations and caveats

* **Single machine vs three.** Each run gives one column group. The paper's
  claims are about relative speedups per machine, which is what one machine
  reproduces.
* **Table 5 is Apple-Silicon-specific.** The paper's numbers are from the M2.
  The scripts run on x86 too, but the loop unrolling that makes VecRec's
  Smith–Waterman kernel fast relies on NEON's 32 vector registers; on x86 the
  gap between VecRec and the tiled scalar code is much smaller.
* **Schedule search is stochastic in time.** `wild`/`leetcode`/`bio` pick the
  fastest schedule by measurement, so a noisy machine can pick a slightly
  different schedule. Run on an idle machine; `--regen` repeats the search.
* **Table 4 absolute times** depend on the dataset download being identical
  (the fetch scripts pin NCBI accessions and a fixed GitHub file) and on
  single-thread CPU speed; the paper's machine for this table is not one of the
  three above (see `08-eval.tex`).
* **Hand-edited tables.** During the camera-ready the paper's tables were
  touched up by hand (row order, names). `fig` reproduces that final layout;
  the underlying result files are what the scripts produce.
* **ksw2 baselines in Table 3 are faster than in the paper on x86.** The
  paper's x86 ksw2 numbers (~360 / ~580 Melem/s) were measured with the ksw2
  driver built as `KSW_SSE2_ONLY` (the SSE2 emulation path). The driver was
  later changed to a fair build (`-march=native`, SSE4.1 fast path), which is
  what this artifact ships; on x86 ksw2 is then 3-6x faster than in the paper.
  The paper's `ksw2_extf2_sse` row is not reproduced: in the current harness
  that kernel is score-only (no traceback), which is not comparable to the
  traceback-timed implementations in the rest of the table, so it is disabled.
  The Apple M2 column is unaffected (ksw2 runs through sse2neon there).
* **Table 4 does not reproduce on every x86 machine.** On the Zen 4 machine
  used for the reference run in §7 the VecRec drop-in is *slower* than stock
  ksw2 (0.84-0.90x on all three datasets) although the stock times match the
  paper's. The cause is the kernel, not the harness: minimap2's gap-fill calls
  are ~225x225 with maximum scores above the int8 dispatch threshold, so they
  run VecRec's int16 kernel, and on this CPU that kernel is ~1.4x slower per
  call than ksw2's int8 difference-encoded `extd2` kernel (isolated
  `bio_kernel_evals/cases/extension_dual_affine.py` at 256x256: ksw2 27 us,
  VecRec int16 37 us, VecRec int8 28 us; the e2e per-call phase timers agree).
  Sweeping the drop-in schedule / vector width (`VECREC_SCHED_2P`,
  `VECREC_MULT_2P`, `VECREC_MAX_BITS`) does not change this; neither does the
  compiler (clang 20 vs 21), nor using the pre-merge VecRec revision. The
  speedups in the paper were measured on an AVX2 (Zen 3) machine where the
  int16 kernel beats ksw2 at these sizes.
* **Table 4's `ecoli-ultralong` row** uses the 500 longest reads of the full
  Loman/Quick ultra-long run (`ecoli-ultralong-full`, a 5 GB download that is
  reduced to a 10 MB subsample); the 371-read `ecoli-ultralong-real` subset is
  ~10x smaller and runs in ~6 s instead of ~60 s.

## 7. Reference run on this artifact's development machine

`results/*/zen4-7985wx.*` are a complete run on an AMD Threadripper PRO 7985WX
(Zen 4, AVX-512), clang 21, with the pinned VecRec. They are not one of the
paper's three machines; they are provided as a known-good example of the
output. Table 3 numbers from this machine agree within a few percent with an
earlier (July 2026) run of the same harness on the same machine.
