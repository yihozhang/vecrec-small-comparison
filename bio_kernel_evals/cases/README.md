# bio_kernel_evals/cases/

Each file defines one `EvalCase` subclass. Cases are auto-discovered by
`bio_kernel_evals/run.py` — just drop a new `.py` here and it appears in the
suite automatically.

## Running cases

```bash
# Run one case (default 128×128, seed=0)
uv run python -m bio_kernel_evals.cases.needleman_wunsch
uv run python -m bio_kernel_evals.cases.affine_gap
uv run python -m bio_kernel_evals.cases.banded_needleman_wunsch

# Run a specific size
uv run python -c "
from evals_common.harness import run_cases
from bio_kernel_evals.cases.affine_gap import AffineGap
run_cases([AffineGap(rows=512, cols=512)])
"

# Full suite with all default sizes
uv run python bio_kernel_evals/run.py

# Full sweep (sizes 64, 128, 256, 512, 1024)
uv run python bio_kernel_evals/run.py --sweep
```

## Case inventory

| File | Algorithm | Baselines | Notes |
|------|-----------|-----------|-------|
| `needleman_wunsch.py` | Global NW (linear gap) | SeqAn3, parasail (striped/diag/scan × 8/16/32/64/sat), ksw2 `gg2_sse`, WFA2 | parasail `_8` variants are timing-only (saturate at N>~64); also `NeedlemanWunschSim65` (WFA2-only, ~65% identity) |
| `smith_waterman.py` | Local SW (linear gap) | SeqAn3, parasail (striped/diag/scan × 8/16/32/64/sat) | global max over full matrix; parasail `_8` variants timing-only |
| `affine_gap.py` | Global affine (Gotoh) | parasail (striped/diag/scan × 8/16/32/64/sat), ksw2 `gg2_sse`/`extz2`/`extd2`, WFA2 | parasail `_8` variants timing-only; also `AffineGapSim65` (WFA2-only, ~65% identity) |
| `edit_distance.py` | Levenshtein | edlib, WFA2 | also `EditDistanceSim65` (WFA2-only, ~65% identity) |
| `banded_needleman_wunsch.py` | Banded NW (bw=128) | SeqAn3 banded, ksw2 `gg2_sse` (correctness), parasail `nw_banded` (correctness, scalar), ksw2 `extz2_sse`/`extf2_sse` (timing-only, banded) | three independent correctness anchors |
| `banded_smith_waterman.py` | Banded SW (bw=128) | SeqAn3 banded | |
| `banded_affine.py` | Banded affine (bw=128) | ksw2 `gg2_sse` (correctness), SeqAn3 `nw_banded` w/ affine gap (correctness), ksw2 `extd2_sse` (timing-only) | two independent correctness anchors |
| `affine_gap_local.py` | Affine-gap LOCAL (restart-at-0), score-only | ksw2 `ll_i16` (correctness) | no linear/global counterpart covers this combination |
| `profile_hmm_viterbi.py` | HMM Viterbi | HMMER3 | requires HMMER built |
| `extension_affine.py` | VecRec 1-piece affine (= minimap2 drop-in kernel) vs ksw2 extension | ksw2 `extz2_sse` + `extz2_sse_approx` (correctness, both banded + unbanded) | synthetic-data anchor for the real minimap2 drop-in kernel |
| `extension_dual_affine.py` | VecRec dual-affine (= minimap2 drop-in kernel) vs ksw2 extension | ksw2 `extd2_sse` + `extd2_sse_approx` (correctness unbanded; banded has no reference) | synthetic-data anchor for the real minimap2 drop-in kernel |
| `example_first_order.py` | Second-order IIR (arithmetic semiring) | numpy | non-alignment demo case |

Note: run the harness (see "Running cases" above) for current throughput numbers
-- this table intentionally does not hardcode speedups, since the baseline set
changes over time and stale numbers are misleading.

## Isolated approx-max A/B (`*_approx` rows)

`extension_affine.py`/`extension_dual_affine.py` add a `ksw2_ext{z,d}2_sse_approx`
row: the identical kernel call with `KSW_EZ_APPROX_MAX` set (minimap2's
`MM_EXACT_ONLY=0` gap-fill shortcut, see `end_to_end_baselines/minimap2/CLAUDE.md`
"approx-max" scenario), isolated at the single-DP-call level (no pipeline/index/
chaining overhead -- `km=nullptr` here, so `kmalloc`/`kfree` are plain system
malloc/free).

This reproduces the real e2e speed gap correctly: a single isolated call already
shows exact ~2x slower than approx-max (e.g. 25.9µs vs 14.2µs at 256x256),
matching the aggregate ~2x DP-phase gap minimap2's own `VECREC_PHASE_TIMING`
instrumentation reports on a real read set. Root cause is algorithmic, not an
allocator effect: `ksw2_extz2_sse.c`'s exact branch does real O(n) extra work
per anti-diagonal row (a 4-wide scalar-gathered int32 max-scan,
`_mm_setr_epi32(v8[t],v8[t+1],v8[t+2],v8[t+3])`, comparable order to the main
16-wide uint8 sweep but at 1/4 the effective SIMD width) plus one extra
`kmalloc`'d `H` array per call, versus the approx branch's O(1) single-column
update and no extra allocation. Scores still match exactly (`delta=0`) --
`approx_max` only skips this bookkeeping, not anything the gap-fill corner
traceback reads.

**Compare against the full VecRec width sweep (I8/I16/I32), not I16 alone.**
`ksw2_extz2_sse_approx` vs. VecRec-I16-only understates VecRec: the I8 kernel
(2x lanes/register) closes most or all of the gap, and outright **wins** in a
mid-size band. On the 1-piece kernel (unbanded, `sequence_inputs` scoring):
approx-max is ~1.0-1.1x faster at 128-256 (near tie), **VecRec-I8 wins by
1.10-1.31x at 512-2048** (still within I8's correct range here), and approx-max
regains the lead by ~1.3x at 4096 once I8 saturates and the fastest correct
schedule falls back to I16. The 2-piece kernel shows the same shape with the
I8 crossover happening a size earlier (I8 saturates by 2048 there, one more
gap state to fit in the same 8-bit range). Always check the sweep across sizes
for current numbers -- the I8 saturation boundary depends on the scoring
scheme and shifts if `match`/`mismatch`/`gap` constants change.

`ksw2_ext{z,d}2_sse_kalloc_batch` (see `baselines/ksw2/align.cpp`) is an
additional, more elaborate isolated check: many calls of varying size (mirroring
the real `[extend-size]` histogram from `MM_COUNT_MATRIX`) sharing ONE
long-lived `km_init()` pool, matching how `map.c`'s `b->km` is actually reused
across a whole thread's reads (`kalloc.c`'s allocator is a linear-search
circular free list, unlike glibc's size-class bins, so its behavior under a
long-lived, multi-size-interleaved pool is worth checking separately from a
single `km=nullptr` call). It reproduces essentially the same ~2x ratio
(e.g. 44.1ms vs 22.7ms over a 259-call batch) -- consistent with the algorithmic
explanation above rather than pointing to a separate allocator-specific cost.

**Gotcha this section exists to warn about**: `cmake --build build/baselines
--target ksw2_align` writes to `build/baselines/bin/ksw2_align`, *not* the
top-level `bin/` the Python harness actually reads
(`evals_common/cpp.py::_bin_dir()`). Rebuilding only via `cmake --build` without
also `cp`-ing (or re-running `scripts/build_baselines.sh`, which does the copy)
silently leaves the harness running the stale old binary -- indistinguishable
from a real "no difference" result if the change you're testing added a new
field the stale binary doesn't parse, since it'll then just ignore it and run
the old code path for every row.

## VecRec element-width sweep (I8 / I16 / I32)

`needleman_wunsch.py`, `smith_waterman.py`, `affine_gap.py`, `edit_distance.py`,
their three banded counterparts, and `extension_affine.py`/`extension_dual_affine.py`
wire VecRec via `evals_common.vecrec_cpp.vecrec_traceback_baselines_multi_width`, which builds
the kernel expression once per `ElementType` (I8, I16, I32) and enumerates
every SIMD schedule for each width, labeling results
`vecrec_<kind>_i8_tb:k*`, `_i16_tb:k*`, `_i32_tb:k*`. Autoscheduling therefore
picks the best **(width, schedule)** pair, not just the best schedule at a
fixed I16 -- narrower widths fit more lanes per physical SIMD register and
can win on throughput (I8 is frequently the fastest width at small N in
practice).

I8 uses saturating arithmetic (clamp, not wrap), so once a score exceeds the
type's range it silently returns a wrong (clamped) value -- this shows up as
a correctness `FAIL` in the table, not a crash. VecRec correctness failures
never gate the suite exit code (they're the subject under test, not a
trusted baseline), so I8 stays visible for timing comparison even past the
size where it stops being correct. Comparing `PASS`/`FAIL` across widths at
each sweep size tells you the largest N where a given width is still safe
for that scoring scheme; run the sweep (`--sweep`, or instantiate a case at a
specific size) to see where each width's ceiling actually falls.

A case that wants the sweep just needs `kernel_expr(self, element_type=
ElementType.I16)` to build its expr at the requested width (instead of a
fixed I16) and to call `vecrec_traceback_baselines_multi_width(self.kernel_expr,
kind=...)` (passing the *method*, not `self.kernel_expr()`) from
`cpp_baselines()`. See `needleman_wunsch.py` for the reference shape.

## Private helper modules

| File | Purpose |
|------|---------|
| `_alignment.py` | Reference DP functions: `nw_reference`, `sw_reference`, `affine_reference`, `banded_*_reference`, `banded_sw_reference` |
| `_hmm.py` | HMM reference Viterbi implementation |
| `_hmm_kernel.py` | VecRec HMM kernel expression |

## Writing a new case

```python
from evals_common.harness import Baseline, EvalCase
from evals_common.vecrec_cpp import vecrec_traceback_baselines

class MyCase(EvalCase):
    name = "my-algorithm"

    def __init__(self, rows=128, cols=128, seed=0):
        self.rows, self.cols, self.seed = rows, cols, seed

    def inputs(self):
        # return {"x": str, "y": str, ...} used by both vecrec and baselines
        ...

    def vecrec_expr(self):
        # return the VecRec SignalExpr to benchmark
        ...

    def optimal_score(self, inputs):
        # return the correct scalar score for correctness checking
        ...

    def baselines(self):
        return [
            Baseline("numpy_ref", lambda **kw: my_numpy_impl(**kw)),
            # or CppBaseline("seqan3_nw", "bin/seqan3_nw", "nw"),
        ]
```

The case is auto-discovered and appears in the suite output immediately.

## Banded cases — key parameter

All banded cases use `BAND_WIDTH = 128`. Always keep `band_width >= 4 × epi`
(at `epi=32` on NEON that means `bw ≥ 128`) to avoid effective-band rounding
artifacts. See root `CLAUDE.md` for details.

Narrower element widths fit more lanes per register, so I8 schedules can have
a larger `epi` than the I16 schedules `BAND_WIDTH` was originally tuned for.
If the widest I8 schedule violates `band_width >= 4 × epi`, it will show a
correctness `FAIL` from band-rounding (not saturation) at that schedule --
expected, and distinguishable from a saturation FAIL by checking whether I16
schedules at the same `k` index also fail (band-rounding hits all widths at
a given schedule shape; saturation is width-specific).
