# Reptile — REPTILE-style tiling of VecRec alignment recurrences

This folder applies the tiling philosophy of **REPTILE** (Tariq, Sundram,
Kjolstad, *Performant Tiling of Recurrences*, OOPSLA 2025) to the two dense-DP
recurrences in this repo: **Needleman-Wunsch** (`bio_kernel_evals/cases/needleman_wunsch.py`)
and **affine-gap / Gotoh** (`bio_kernel_evals/cases/affine_gap.py`).

The approach is to **treat each VecRec inner-loop kernel as a black box (an
"external function" in REPTILE terms) and construct tile loops around it.**
The tiled scalar loop nests are taken from the implementation in REPTILE's
artifact. They are embedded as code templates in `reptile.py` rather than
imported as an external dependency, so they are exactly what REPTILE implements.
The tile loops *around* the VecRec kernel are this repo's composition of
REPTILE's tiling with VecRec, also emitted by `reptile.py`. This folder does not
re-implement the VecRec/RECUMA compiler.
`bench.py` builds and runs them across tile sizes, tiling approaches, and matrix
sizes. Column and 2D tiling of the real kernel rely on one small, general
primitive in vec-rec — checkpoint/restore of the recurrence carries at a tile
boundary (`save_col_state`/`restore_col_state` + `seek_row`).

## How REPTILE maps onto this repo

| REPTILE concept | Here |
|---|---|
| Recurrence equation (input) | NW: `S(i,j)=max(S(i,j-1)+g, S(i-1,j)+g, S(i-1,j-1)+s)`; affine Gotoh (M/Dx/Dy) |
| Tiling variable + tile size | `approach` ∈ {row, col, block} + `tile_rows`/`tile_cols` (read at run time) |
| External function mapping | The compiled **VecRec inner-loop kernel** (`kernel.run/seek_col/seek_row/reset_and_next_row` + `save_col_state`/`restore_col_state`) is the mapped tile primitive |
| Compiler-generated loops (no mapping) | REPTILE's tiled scalar DP loop nest for the same recurrence, taken from REPTILE's artifact |
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
   All four tiling shapes run against the *real* generated kernel:
   * `untiled`: one full row-by-row sweep of the black box.
   * `row`: REPTILE **row-strip tiling** — an outer loop over row-tiles, each
     computed by the same continuous kernel state.
   * `col`: REPTILE **column-panel tiling** — full-height panels of width
     `tile_cols`, visited left-to-right.
   * `block`: REPTILE **2D tiling** — row-band outer, column-panel inner
     (`tile_rows × tile_cols` blocks).

   Column/2D tiling of the real kernel needs extra support because the
   generated kernel carries its recurrence state (up/diagonal via a full-width
   `RepeaterContext` ring buffer; the in-row left dependency via `SRecurse`;
   the diagonal shift via `SConvolve`) *continuously* along each row, so
   restarting mid-matrix at column `cj0 > 0` lost the left/diagonal carries —
   and for affine those carries (`E`, `F`) are **not** recoverable from the
   written `H` output. vec-rec provides a small, general **checkpoint/restore**
   primitive so a tiled driver can restart the recurrence anywhere:

   | Boundary crossed | State that must survive | Primitive |
   |---|---|---|
   | left edge of a column panel | `SConvolve` diagonal-shift carry + `SRecurse` in-row (left / affine `F`) carry | `kernel.save_col_state()` / `restore_col_state()` (checkpoint the *actual* internal carry, so it works even when the value isn't in the output) |
   | row phase after a restart | logical row counter (profile row + ring slot) | `kernel.seek_row(r)` |
   | top edge of a 2D block | previous row (`H`, and affine `E`) | *none needed* — the full-width ring buffer already holds each column's previous-row value, because bands run top-to-bottom and panels within a band cover disjoint columns |

   The driver reuses **one** kernel: at each column panel it `seek_row`s to the
   band top and, for `cj0 > 0`, restores the per-row carries saved at the right
   edge of the previous panel. Because we checkpoint the real internal state
   rather than reconstructing it from `H`, this is exactly correct for the
   collapsed 2-state affine kernel too (verified bit-identical — see below).

2. **`loops` family — REPTILE's tiled scalar loops (no external mapping).**
   The tiled scalar loop nest from REPTILE's artifact, for a recurrence whose
   tiles map to no library. Approaches:
   `untiled` / `row` / `col` / `block`, sweeping tile size.

## Run it

From the repo root (needs clang ≥ 18, which VecRec requires):

```bash
# Default: tiled VecRec vs untiled VecRec (`vecrec` family), both recurrences:
uv run python Reptile/bench.py

# Both families (adds the tiled scalar `loops` family):
uv run python Reptile/bench.py --family both

# Narrower sweep:
uv run python Reptile/bench.py --family both --kinds nw --sizes 512,1024 --tiles 64,256 --reps 3
```

`--family` selects which driver family(ies) to build and sweep: `vecrec`
(default), `loops`, or `both`. The harness:
1. compiles the VecRec external-mapping driver for **every candidate schedule**
   and picks one via a quick untiled timing probe, when `family` includes
   `vecrec`;
2. compiles the REPTILE-generated tiled-loops driver, when `family` includes `loops`;
3. sweeps `{size} × {approach} × {tile size}`, validates each score, and prints
   median runtime relative to the untiled baseline of each family;
4. with `--family both`, also reports, per size, the best scalar-loops config
   against plain VecRec untiled (saved as `vecrec_vs_scalar` in the JSON).

Tables print to stdout; each run also writes `results_<stamp>.json` and
`summary_<stamp>.md` into `Reptile/results/` (created on first run).
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

The affine VecRec `col`/`block` variants are the key test of the
checkpoint/restore: affine's `E`/`F` gap states never appear in the written `H`
matrix, so a tiling that reconstructed carries from `H` would be wrong. Note that
the VecRec and `loops` affine matrices share the same corner score but can differ
in interior checksum, because VecRec's collapsed 2-state affine uses a different
boundary convention (documented in `bio_kernel_evals/cases/_alignment.py`); the
identity check therefore compares within a family, not across families.

## Files

| File | Purpose |
|---|---|
| `reptile.py` | REPTILE-lite codegen: VecRec black-box header gen + scalar `loops` driver gen + both tiled driver families (incl. `col`/`block` via checkpoint/restore) + clang build |
| `bench.py` | Sweep harness: `--family {vecrec,loops,both}` selection, schedule selection, `{size×approach×tile}` sweep, validation, cross-family comparison, tables, artifacts |
| `perf_pressure.py` | Runs tiled vs untiled VecRec drivers under `perf stat` to collect hardware cache/memory counters |
| `roofline_score.py` | Score-only tiled vs untiled VecRec runs at very large N; checks that all traversals report the same score |
| `results/` | JSON + Markdown output written by `bench.py` (created at run time) |
| `build/` | Compiled drivers (gitignored) |

The column/2D tiling primitives live in the vec-rec package (shared, so all
kernels get them). They are already part of the pinned `vec-rec` submodule
(vec-rec commit `197101f`, "allow for proper tiling"); no separate patch needs
to be applied. The additions are:

| vec-rec file | Addition |
|---|---|
| `templates/common.h` | `save_state`/`restore_state` on `SConvolve*`/`SRecurse*`/`SRecurseV*`; `prime_left_carry` on `SRecurse*`; `seek_row` + `save_prev_row`/`load_prev_row` on `RepeaterContext`; `seek_row` on `ProfileSignal2D`/`InputSignal2D` |
| `codegen/codegen.py` | generated kernel gains `save_col_state`/`restore_col_state`/`col_state_bytes`, `seek_row`, `num_ring_buffers`, `save_ring_row`/`load_ring_row` |
