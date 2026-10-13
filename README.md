# VecRec — Artifact Evaluation

This repository reproduces the evaluation of the VecRec paper. 
In this paper, we build a compiler for vectorizing linear recurrences over arbitrary semirings
 and demonstrated significant speedups across various domains.
In this artifact evaluation, we provide the code and scripts to reproduce the results in the paper.
VecRec itself is implemented as a Python library that generates a C++ kernel file that can be imported by other C++ programs.
It is pinned as the git submodule [`vec-rec/`](vec-rec).

| Paper table | Group name | What is measured |
|---|---|---|
| Table 1 | `wild` | 12 recurrences in the wild (e.g., signal processing / numerics). The baselines are hand-written serial C++ loops and VecRec kernels without vectorization |
| Table 2 | `leetcode` | 12 LeetCode dynamic-programming recurrences. The baselines are hand-written serial C++ loops and VecRec kernels without vectorization. |
| Table 3 | `bio` | Sequence alignment algorithms, including edit distance, Needleman–Wunsch, Smith–Waterman, and affine gap. The baselines include parasail, ksw2, seqan3, edlib, WFA2 |
| Table 4 | `e2e` | End-to-end performance comparison on the three real ONT datasets. The baseline is the unmodified ksw2 implementation. |
| Table 5 | `reptile` | We compare with REPTILE, a recent tiling compiler for recurrences, on the Smith–Waterman algorithm. |

To run this artifact, you can use the following commands:

```bash
uv run bench exec <group>     # run the experiments behind one table  (results/<group>/<host>.csv)
uv run fig <group>            # render that table as LaTeX + Markdown  (figures/<group>.{tex,md})
```

In the paper, we reported the benchmark results on three microarchitecutres, including  Intel Skylake (i9-9960X), AMD Zen 5 (Threadripper 9970X), Apple M2. You may see different performance characteristics on a different micro-architecture.

The script provided in this repository can be used to obtain result on one machine, but you can use `fig` to assemble a table with results from multiple machines side by side.

We have added numbers we obtained by running this artifact on a few microarchitecutres, they are named `results/<group>/<cpu-name>.{csv,json}`.

## 1. Requirements

This artifact requires the following setups:

* Linux x86-64 or macOS/arm64. A machine with a fairly recent vectorization instruction set (e.g.m AVX-512 or NEON). All experiments are single-threaded. We have not tested this artifact on a machine with the AVX2 instruction set only, but we expect the observed speedup to be slightly smaller. On Linux, you can use `lspu` or `cat /proc/cpuinfo` to view the supported instruction set.
* **clang ≥ 17** on `PATH` (VecRec kernels use clang's vector extensions. seqan3 also needs clang ≥ 17). On macOS use Homebrew LLVM (`brew install llvm`), not Apple's.
* [`uv`](https://docs.astral.sh/uv/) (Python ≥ 3.10 is fetched by uv if needed).
* `git`, `make`, `cmake ≥ 3.20`, `curl`, `gunzip`, and zlib development headers
  (`zlib1g-dev` on Debian/Ubuntu; present by default on macOS). This is needed by the
  third-party aligners in Tables 3–5.
* Disk: ≈ 5 GB (third-party clones + builds).

## 2. Setup

```bash
git clone --recursive https://github.com/yihozhang/vecrec-small-comparison
cd vecrec-small-comparison
uv sync                          # creates .venv with vecrec (editable) + numpy
uv run bench list                # sanity check: prints groups and the host tag
```

If you cloned without `--recursive`, run `git submodule update --init` first.

## 3. Kicking the tire

To sanity check, you can run the following command to obtain the run time for a single benchmark:

```bash
uv run bench exec wild --only second-order-low-pass
```

You should see the detailed statistics of the second-order low-pass filter comparison. Here is an example:

```
===== bench exec wild (Table 1) on host 'griffin' =====

[gen] second-order-low-pass: reusing generated/second_order_low_pass.h
make: 'build/second_order_low_pass' is up to date.

==== second-order-low-pass ====
  elements                : 4194304
  serial          (best)  :     5541.2 us    0.757 Gelem/s
  VecRec baseline (best)  :     7904.4 us    0.531 Gelem/s
  VecRec optimized(best)  :     1210.5 us    3.465 Gelem/s
  speedup  opt  vs serial : 4.58x
  speedup  opt  vs base   : 6.53x
  speedup  base vs serial : 0.70x
  max abs err base / opt  : 1.764e-05 / 2.289e-05
  correctness             : PASS
RESULT,second-order-low-pass,4194304,5541.183,7904.447,1210.477,4.5777,6.5300,0.7010,1.764e-05,2.289e-05,PASS

benchmark                                 serial us    base us     opt us  opt/ser  opt/base  status
second-order-low-pass                        5541.2     7904.4     1210.5    4.58x     6.53x    PASS
```

To get a sense of how to generate recurrence kernels using VecRec, you can take a look at programs under `vec-rec/benchmarks/`. To run a kernel there, you can use the command `uv run vec-rec/benchmarks/run_all.py KERNEL_NAME`. For example,

```
uv run vec-rec/benchmarks/run_all.py first-order-high-pass
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

Raw results will be saved at `results/<group>/<host>.csv` (`.json` for `reptile`), where
`<host>` is obtained from `--host NAME`, `$AE_HOST`, or the hostname by default. Re-running a group overwrites previous results.

Useful flags:

* `--only NAME` — one benchmark (`wild`/`leetcode`: registry name such as
  `second-order-low-pass`; `bio`: case name such as `affine-gap`;
  `e2e`: dataset label such as `pa-strains`). Repeatable.
* `--regen` (`wild`/`leetcode`) — redo the VecRec schedule search to find the best-performaning kernel instead of
  reusing `generated/<kernel>.h` from a previous run.
* `--reps N` (`e2e`, `reptile`), `--size N` (`bio`), `--iters/--warmup` (`wild`/`leetcode`).
* `--quick` — smoke-test settings (tiny sizes, few iterations). Finishes in
  minutes and exercises every code path.

To start over, run `uv run bench clean <group>` to delete build cache. The flag `--deep` also removes `third_party/` and the
downloaded datasets. Result files under `results/` are never touched.

To render a table from an experiment:

```bash
uv run fig wild                  # prints a Markdown table, writes figures/wild.{tex,md}
uv run fig all
```

`fig` uses every file in `results/<group>/` as one column group (labelled by
file stem). To choose files and labels explicitly:

```bash
uv run fig bio --source "Zen 5=results/bio/zen5.csv" --source "Apple M2=results/bio/m2.csv"
```

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
`ecoli-ultralong` = the 500 longest reads of the full Loman/Quick ultra-long ONT *E. coli* run vs K-12 MG1655;
`ecoli-strains` = *E. coli* O157:H7 chopped into 100 × 50 kbp contigs vs K-12;
`pa-strains` = *P. aeruginosa* PA14 contigs vs PAO1.

### Table 5 (`reptile`)

[`Reptile/`](Reptile): REPTILE's tiled scalar loop nest for Smith–Waterman,
taken from the implementation in REPTILE's artifact (embedded as code templates in
`Reptile/reptile.py` rather than imported as an external dependency), under
untiled / row / column / 2-D block tiling with tile sizes 4–512, and
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

## 6. Other caveats

* Make sure your machine is quiet during the evaluation to make sure the numbers are reliable.
* Table 2: You may observe a speedup of ksw2's throughputs over what is reported in the paper, as 
  we added the `-march=native` flag to ksw2 to make it to use the available vectorizing instruction set.
  It still has a lower throughput than the best-performing baseline on the machines we have tested, however.
* Table 4: You may not observe a speedup depending on the machine you use. The reported speedups
  can be reproduced on a machine with Zen 5 microarchitecture. We are happy to provide AE reviewers
  access to our machine for replication.
* Table 5: The speedup of REPTILE over VecRec can only be reproduced on a machine with Apple Silicon. 
  This is because tiling, or loop unrolling, only improves Smith-Waterman on microarchtectures
  with massive instruction-level parallelism, and on microarchitectures like Zen 5 the kernel
  is already register pressured, so unrolling does not help. However, you should be able to observe
  the speedup of VecRec over REPTILE and Untiled baselines.
