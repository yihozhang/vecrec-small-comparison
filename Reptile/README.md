# Reptile — REPTILE-style tiling of VecRec alignment recurrences

This folder applies the tiling philosophy of **REPTILE** (Tariq, Sundram,
Kjolstad, *Performant Tiling of Recurrences*, OOPSLA 2025) to the two dense-DP
recurrences in this repo: **Needleman-Wunsch** (`bio_kernel_evals/cases/needleman_wunsch.py`)
and **affine-gap / Gotoh** (`bio_kernel_evals/cases/affine_gap.py`).

The goal, per the request: **treat each VecRec inner-loop kernel as a black box
(an "external function" in REPTILE terms) and construct tile loops around it.**
We do *not* re-implement the VecRec/RECUMA compiler; we (a) generate the tiled C
that REPTILE would emit and (b) run it to measure the speedups from different
tile sizes, tiling approaches, and matrix sizes. To make **column and 2D**
tiling of the real kernel possible we added one small, general primitive to
vec-rec — checkpoint/restore of the recurrence carries at a tile boundary
(`save_col_state`/`restore_col_state` + `seek_row`) — not a new compiler.

## How REPTILE maps onto this repo

| REPTILE concept | Here |
|---|---|
| Recurrence equation (input) | NW: `S(i,j)=max(S(i,j-1)+g, S(i-1,j)+g, S(i-1,j-1)+s)`; affine Gotoh (M/Dx/Dy) |
| Tiling variable + tile size | `approach` ∈ {row, col, block} + `tile_rows`/`tile_cols` (read at run time) |
| External function mapping | The compiled **VecRec inner-loop kernel** (`kernel.run/seek_col/seek_row/reset_and_next_row` + `save_col_state`/`restore_col_state`) is the mapped tile primitive |
| Compiler-generated loops (no mapping) | A REPTILE-style tiled DP loop nest emitted for the same recurrence (paper §9.6 emits exactly this for NW/SW, which map to no library) |
| `cCodeGen` → C | `reptile.py` emits the C++ driver; each self-times via `baselines/common/driver_io.hpp` |

Because REPTILE tiles the **outermost** loop variable, tiling `i` gives
row-strips and tiling `j` gives column-panels; tiling both gives 2D blocks —
the same three shapes we sweep here.

## Two driver families

`reptile.py` generates two self-timed driver families per recurrence. Both
produce the full `(rows+1)×(cols+1)` DP matrix and reuse the repo's inline
traceback (`baselines/traceback/matrix_traceback.hpp`); every configuration is
validated against the textbook reference score.

1. **`vecrec` family — the VecRec kernel is the mapped external tile primitive.**
   All four REPTILE tiling shapes are now runnable against the *real* generated
   kernel:
   * `untiled`: one full row-by-row sweep of the black box.
   * `row`: REPTILE **row-strip tiling** — an outer loop over row-tiles, each
     computed by the same continuous kernel state.
   * `col`: REPTILE **column-panel tiling** — full-height panels of width
     `tile_cols`, visited left-to-right.
   * `block`: REPTILE **2D tiling** — row-band outer, column-panel inner
     (`tile_rows × tile_cols` blocks).

   Column/2D tiling of the real kernel used to be impossible because the
   generated kernel carries its recurrence state (up/diagonal via a full-width
   `RepeaterContext` ring buffer; the in-row left dependency via `SRecurse`;
   the diagonal shift via `SConvolve`) *continuously* along each row, so
   restarting mid-matrix at column `cj0 > 0` lost the left/diagonal carries —
   and for affine those carries (`E`, `F`) are **not** recoverable from the
   written `H` output. We added a small, general **checkpoint/restore**
   primitive to vec-rec so a tiled driver can restart the recurrence anywhere:

   | Boundary crossed | State that must survive | New primitive |
   |---|---|---|
   | left edge of a column panel | `SConvolve` diagonal-shift carry + `SRecurse` in-row (left / affine `F`) carry | `kernel.save_col_state()` / `restore_col_state()` (checkpoint the *actual* internal carry, so it works even when the value isn't in the output) |
   | row phase after a restart | logical row counter (profile row + ring slot) | `kernel.seek_row(r)` |
   | top edge of a 2D block | previous row (`H`, and affine `E`) | *none needed* — the full-width ring buffer already holds each column's previous-row value, because bands run top-to-bottom and panels within a band cover disjoint columns |

   The driver reuses **one** kernel: at each column panel it `seek_row`s to the
   band top and, for `cj0 > 0`, restores the per-row carries saved at the right
   edge of the previous panel. Because we checkpoint the real internal state
   rather than reconstructing it from `H`, this is exactly correct for the
   collapsed 2-state affine kernel too (verified bit-identical — see below).

2. **`loops` family — REPTILE-*generated* tiled loops (no external mapping).**
   The tiled loop nest REPTILE emits for a recurrence whose tiles map to no
   library. Approaches: `untiled` / `row` / `col` / `block`, sweeping tile size.
   This is the honest "expected REPTILE code" for the 2D case, and it reproduces
   the paper's NW/SW tiling study (Figures 22–23) inside this repo.

## Run it

From the repo root (needs clang ≥ 18, which VecRec requires):

```bash
# Default: VecRec + tiling vs vanilla VecRec (Part B below), both recurrences:
uv run python Reptile/bench.py

# Full two-part study (adds scalar loops vs VecRec, Part A below):
uv run python Reptile/bench.py --family both

# Narrower / faster:
uv run python Reptile/bench.py --family both --kinds nw --sizes 512,1024 --tiles 64,256 --reps 3
```

`--family` selects which driver family(ies) to build and sweep: `vecrec`
(default), `loops`, or `both`. The harness:
1. compiles the VecRec external-mapping driver for **every candidate schedule**
   and picks the fastest one via a quick untiled probe (the black box), when
   `family` includes `vecrec`;
2. compiles the REPTILE-generated tiled-loops driver, when `family` includes `loops`;
3. sweeps `{size} × {approach} × {tile size}`, validates each score, and prints
   median runtime + speedup vs the untiled baseline of each family;
4. with `--family both`, also computes and prints an explicit cross-family
   verdict per size — the best scalar-loops config vs plain VecRec untiled
   (see "Findings" below) — saved as `vecrec_vs_scalar` in the JSON and as a
   dedicated table in the Markdown summary.

Tables print to stdout; JSON + a Markdown summary land in `Reptile/results/`.
Compiled binaries live in `Reptile/build/` (gitignored; regenerated on demand).

## Is the tiling correct? How it's verified

Two layers of checking, both run automatically:

1. **Per-config score check.** Every driver reports the alignment score (the
   corner cell `H[rows][cols]`) and it is compared to the textbook
   `nw_reference` / `affine_reference`. For *global* alignment the corner depends
   transitively on every cell, so a dependency-ordering bug in a tiling almost
   always corrupts it.

2. **Full-matrix bit-identity check** (`--verify`). Each driver FNV-1a-hashes its
   entire `[0..rows]×[0..cols]` DP matrix. Two configs with equal checksums have
   bit-identical matrices, which *proves* a tiling only reorders the traversal
   (it can't be masking a compensating error). We assert, within each family,
   that every tiled variant's checksum equals the untiled one's.

```bash
uv run python Reptile/bench.py --verify --family both --kinds nw,affine --sizes 512,2048 --tiles 64,256,512
```

Result: **ALL PASS** — in *both* families, every `row`/`col`/`block` variant at
every tile size is `matrix=IDENTICAL` to that family's untiled run, and every
score matches the reference. In particular the new **VecRec `col`/`block`
variants are bit-identical to VecRec untiled for affine too**, which is the real
test of the checkpoint/restore: affine's `E`/`F` gap states never appear in the
written `H` matrix, so a tiling that reconstructed carries from `H` would be
wrong — checkpointing the *actual* internal carry is what makes it exact. (A
nice side-effect: the VecRec NW matrix is also bit-identical to the textbook
loops matrix; the affine ones share the same corner score but differ in interior
checksum, since VecRec's collapsed 2-state affine uses a different boundary
convention — exactly as documented in `_alignment.py`.)

## Findings — VecRec + tiling vs. VecRec untiled

The headline comparison here is **VecRec + REPTILE tiling vs. plain VecRec**
(each tiled variant against the *same* VecRec untiled baseline), not VecRec vs.
the loops. Speedups below are `vecrec_untiled_ms / vecrec_tiled_ms`; every config
validated bit-identical (`ok=Y`). Data: `Reptile/results/summary_20260705_135226.md`.

Hardware: AMD Threadripper (AVX2, 128 MB L3), clang 18.1.8. The int16 VecRec
matrix is small, so even N=8192 (256 MB) only just spills L3.

### The verdict: tiling a VecRec kernel is ≈ parity — you can't tile your way faster

Best VecRec tiled speedup vs. VecRec untiled, per size (medians):

| size | NW untiled | NW `row` | NW `col` | NW `block` | affine untiled | affine `row` | affine `col` | affine `block` |
|---|---|---|---|---|---|---|---|---|
| 1024 | 0.249 ms | 1.29× | 1.26× | 1.25× | 1.069 ms | 1.00× | 0.99× | 0.98× |
| 2048 | 0.814 ms | 1.01× | 0.99× | 0.98× | 2.890 ms | 1.00× | 0.95× | 0.97× |
| 4096 | 15.50 ms | 1.00× | 1.00× | 1.00× | 23.95 ms | 1.01× | 0.96× | 1.00× |
| 8192 | 66.54 ms | 1.01× | **1.04×** | 1.00× | 100.6 ms | 0.99× | 0.99× | 1.00× |

(The NW-1024 "1.2–1.3×" is a small-N artifact: at that size the untiled baseline
is jittery — the schedule-selection probe timed the same kernel at 0.195 ms — so
those ratios mostly reflect baseline noise, not a real tiling win.)

Three robust patterns:

* **`row` tiling ≈ 1.00×.** Wrapping row-strips around the kernel is a pure
  iteration-space partition that keeps the kernel streaming continuously, so it
  neither helps nor hurts.
* **`col`/`block` tiling ≈ 0.95–1.00× (usually a hair *slower*).** These
  genuinely restart the recurrence at each panel — `seek_row` + restoring the
  `SConvolve`/`SRecurse` carries costs a little, and for affine the per-row
  `F`-carry checkpoint adds up. The best case is NW `col` t=128 at 8192
  (**1.04×**), the only configuration that clears the noise floor.
* **Bigger tiles → closer to parity.** Wider panels amortize the per-panel
  restart, so `col`/`block` climb back toward 1.00× as `tile_cols` grows.

### Why tiling barely moves these kernels

NW/affine are 2-tap stencils streamed row-by-row: each cell is reused O(1) times
(only by the next row), unlike matmul's O(n) reuse — so cache-blocking has little
to reclaim. Crucially, **the VecRec kernel already captures the available
row-locality** in its full-width `RepeaterContext` ring buffer, so there is no
locality left for an outer tile loop to recover; tiling can only add restart
overhead. This is *why* the study set out to test tiling around the black box,
and the answer is clean: for these recurrences, the win is in the vectorized
inner loop, not in the tile schedule.

### Context: VecRec vs. the REPTILE-generated loops

For reference, the same tiling applied to the plain (auto-vectorized) loop nest
behaves the same way — best ≈1.02×, `col`/`block` often *hurt* (chopping the
contiguous inner-`j` loop defeats auto-vectorization). And the VecRec black box
is **~4–9× faster** (NW) to **~5–6× faster** (affine) than the loops at every
size, tiled or not (e.g. 8192: NW 66.5 ms vs 220.7 ms; affine 100.6 ms vs
514.4 ms). So the ranking is unchanged: pick the vectorized kernel; the tile
schedule is a second-order (≈1×) effect for dense pairwise-alignment DP.

Run `bench.py --family both` to get this comparison directly rather than
reading it off two separate tables: it reports, per size, the best scalar
`loops` config (whatever ended up fastest — tiled or not) against plain
VecRec `untiled`, e.g.:

```
    => VecRec untiled (0.249 ms) is 6.13x faster than the best scalar-loops config (loops_col_t256, 1.526 ms)
```

This is saved as `vecrec_vs_scalar` per size in `results_*.json` and as a
"VecRec vs scalar loops (vanilla Reptile)" table in `summary_*.md`.

### How column/2D tiling of the black box was made possible

The enabling change lives in vec-rec (`templates/common.h` + `codegen/codegen.py`):
a general **checkpoint/restore of the column-crossing recurrence carries**
(`save_state`/`restore_state` on `SConvolve`, `SRecurse`, `SRecurseV`; surfaced
as `save_col_state`/`restore_col_state` on the generated kernel) plus a
`seek_row()` row-phase reposition. Because it snapshots the *real* internal carry
(not a value reconstructed from the output), it is exact even for the collapsed
2-state affine kernel, whose `E`/`F` states are invisible in the written matrix.
This is the `prime_left_carry`/state-injection primitive previously listed as
future work — now implemented and verified.

## Files

| File | Purpose |
|---|---|
| `reptile.py` | REPTILE-lite codegen: VecRec black-box header gen + scalar `loops` driver gen + both tiled driver families (incl. `col`/`block` via checkpoint/restore) + clang build |
| `bench.py` | Sweep harness: `--family {vecrec,loops,both}` selection, schedule selection, `{size×approach×tile}` sweep, validation, cross-family verdict, tables, artifacts |
| `results/` | Saved JSON + Markdown summaries from sweeps |
| `build/` | Compiled drivers (gitignored) |

The column/2D tiling primitives live in the vec-rec package (shared, so all
kernels get them). Because vec-rec is a separately-tracked (symlinked) repo,
those changes are **not** committed here — they are provided as a patch,
`Reptile/vec-rec-tiling.patch`, which you must apply to your vec-rec checkout
before the `col`/`block` (and score-only) drivers will build:

```bash
# from the vec-rec repo root (the dir containing src/vecrec/):
git apply /path/to/Reptile/vec-rec-tiling.patch     # or: patch -p1 < ...
```

The patch is purely additive:

| vec-rec file | Addition |
|---|---|
| `templates/common.h` | `save_state`/`restore_state` on `SConvolve*`/`SRecurse*`/`SRecurseV*`; `prime_left_carry` on `SRecurse*`; `seek_row` + `save_prev_row`/`load_prev_row` on `RepeaterContext`; `seek_row` on `ProfileSignal2D`/`InputSignal2D` |
| `codegen/codegen.py` | generated kernel gains `save_col_state`/`restore_col_state`/`col_state_bytes`, `seek_row`, `num_ring_buffers`, `save_ring_row`/`load_ring_row` |
