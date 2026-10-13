# minimap2 VecRec integration

A VecRec-generated affine kernel replaces minimap2's ksw2 extension functions
(`ksw_extz2_sse`, `ksw_extd2_sse`) by patching `libminimap2.a`. The splice kernel
(`ksw_exts2_sse`) is left stock. Two binaries are produced from one source tree:

| Binary | Kernel |
|--------|--------|
| `bin/minimap2_original` | stock ksw2 |
| `bin/minimap2_vecrec`   | VecRec drop-in |

This doc consolidates: the supported regime + harness layout, the DP recurrence math for the
ksw2 extend kernels, the VecRec integration mechanism, and the compile-flag /
instrumentation reference.

## Supported regime

The VecRec drop-in computes the full extension DP exactly. It is run in the
**fair regime** (no band, no z-drop, no approx-max), in which both binaries do
the same DP work:

- **no banding** — the whole R×C matrix is computed (`-r` wide);
- **no z-drop** — `ez->zdropped` is always 0 (runs use `-z 1000000`);
- **no approx-max** — both binaries compute the exact max on every call
  (`MM_EXACT_ONLY=1`, the build default);
- **CIGAR traceback** always produced (no score-only).

Regime assertions (below) enforce this so no run silently leaves the regime.

## Layout

Everything minimap2-specific lives under `end_to_end_baselines/minimap2/`:

| Path | Role |
|------|------|
| `minimap2/` | committed v2.28 source (instrumentation + regime asserts baked in) |
| `generated/` | codegen output: `vecrec_affine_kernel.h` + `ksw2_vecrec_ext.cpp` |
| `flags.py` | compile-flag registry (name → build env bucket, applies-to, desc) |
| `datasets.py` | dataset registry (synthetic vs real; `fair` marks datasets run in the regime above) |
| `make_datasets.py` | regenerate the synthetic datasets |
| `scenarios.py` | scenario registry (dataset + `-r`/`-z`/preset + defines + metrics) |
| `run.py` | scenario-driven runner (build + parity + timing) |

The build script `end_to_end_baselines/scripts/build_minimap2.sh` and
`generate_kernel.py` still live under `scripts/` (used by `run.py`).

## Quickstart (scenario runner)

```bash
uv run python end_to_end_baselines/minimap2/run.py --list        # list scenarios
uv run python end_to_end_baselines/minimap2/run.py exact-sv       # parity + timing
uv run python end_to_end_baselines/minimap2/run.py exact-smoke    # fast parity gate
```

`run.py` builds the stock + vecrec binaries with the scenario's compile defines
(reused if the last build used the same define-set), then collects the
scenario's metrics: `parity` (per-read SAM agreement) and/or `timing` (median
wall-clock + speedup). Add a comparison by adding a `Scenario` to `scenarios.py`.

Low-level build/compare (what `run.py` drives):

```bash
bash end_to_end_baselines/scripts/build_minimap2.sh              # -> bin/minimap2_{original,vecrec}
uv run python end_to_end_baselines/scripts/compare_e2e_sam.py --zdrop 1000000
```

- VecRec wrapper objects compile with **clang++** (`ext_vector_type`,
  `__builtin_elementwise_*`); C sources use `cc`. x86_64 only.
- Asserts are active in both `-O3` and `-O0` builds (no `-DNDEBUG`).
- **Always run with `-z 1000000`** — the drop-in has no z-drop and asserts enforce it.

Mechanism:
1. `generate_kernel.py` emits `generated/vecrec_affine_kernel.h` (dual-affine +
   1-piece kernels).
2. `build_minimap2.sh` builds the committed minimap2 → `libminimap2.a`.
3. Compiles `generated/ksw2_vecrec_ext.cpp` with `-DUSE_VECREC -DHAVE_KALLOC`.
4. `ar d libminimap2.a <ksw2 objects>; ar r libminimap2.a ksw2_vecrec_ext.o`.
5. Re-links → `bin/minimap2_vecrec`.

---

# minimap2 extend kernels — DP recurrences and encodings

## Coordinate system

All three extend functions sweep **anti-diagonals** r = 0 … qlen+tlen-2. Within
anti-diagonal r, each position is indexed by target coordinate t. The
corresponding matrix cell is

    (i, j) = (r - t, t)      i = query row, j = target column

The outer loop in the C code is `for (r …)`. All three kernels share this
geometry; they differ only in how many gap states they carry and in the
arithmetic encoding.

## Baseline Gotoh recurrence (what all three compute)

Standard 1-piece affine (Gotoh):

    H(i,j) = max( H(i-1,j-1) + s(i,j),  E(i,j),  F(i,j) )
    E(i,j) = max( H(i-1,j) − p,  E(i-1,j) − e )    // gap extending along target
    F(i,j) = max( H(i,j-1) − p,  F(i,j-1) − e )    // gap extending along query
    p = q + e   (gap-open cost for a gap of length 1)

2-piece affine (extd) adds a second pair (q2, e2) and four gap states E, E2, F, F2.

## ksw_extz / ksw_extz2_sse — 1-piece affine, unsigned int8 encoding

**Source**: `ksw2_extz2_sse.c` (minimap2 default for DNA alignment)

State variables stored per cell (i,j), all non-negative (uint8):

| variable | definition |
|----------|-----------|
| `U[r,t]` | `H(i,j) − H(i,j−1) + p` |
| `V[r,t]` | `H(i,j) − H(i−1,j) + p` |
| `X[r,t]` | `max( F(i,j+1) + p − H(i,j),  0 )` |
| `Y[r,t]` | `max( E(i+1,j) + p − H(i,j),  0 )` |

X and Y are "look-ahead" gap states: how much the **next** gap state would
exceed the current H, shifted up by p to stay ≥ 0. When no gap is active they
collapse to 0.

Intermediate quantity Z:

    Z[r,t]  =  H(i,j) − H(i−1,j−1) + 2p

Update equations (from `__dp_code_block1` + `__dp_code_block2`):

```
// three candidates for Z
z  = s(i,j) + 2p                            // substitution (block1, first line)
a  = X[r-1, t-1] + V[r-1, t-1]             // F-gap path (block1)
b  = Y[r-1, t]   + U[r-1, t]               // E-gap path (block1)

Z  = max(z, a, b)
Z  = min(Z, s_match + 2p)                  // uint8 safety clamp

U[r,t] = Z − V[r-1, t-1]                   // block2, line 1
V[r,t] = Z − U[r-1, t]                     // block2, line 2

X[r,t] = max( a − (Z − q),  0 )            // block2 + store
Y[r,t] = max( b − (Z − q),  0 )
```

Why each path recovers the right H:

- **Substitution** (`Z = s+2p`): `U+V[prev] = Z` → `H(i,j)−H(i,j-1)+p + H(i,j-1)−H(i-1,j-1)+p = s+2p` → `H(i,j)=H(i-1,j-1)+s` ✓
- **F-gap** (`Z = a`, X[r-1,t-1] > 0): `a = F(i,j)+p−H(i,j-1) + H(i,j-1)−H(i-1,j-1)+p = F(i,j)−H(i-1,j-1)+2p` → `H(i,j)=F(i,j)` ✓
- **E-gap** (`Z = b`, Y[r-1,t] > 0): analogously `H(i,j)=E(i,j)` ✓

H reconstruction (Z-drop tracking): `H[t] += v8[t] - qe;` confirming
`V[r,t] = H(i,j) − H(i-1,j) + p`.

Arithmetic: unsigned int8 (`_mm_max_epu8`). Works on SSE2 without SSE4.1 for the
score-only path. The clamp at `s_match + 2p` is provably non-binding when the
precondition `−min_score ≤ 2(q+e)` holds; under that condition the encoding is exact.

## ksw_extd / ksw_extd2_sse — 2-piece affine, signed int8 encoding

**Source**: `ksw2_extd2_sse.c`

Carries four gap-state arrays: (X, Y) for penalty pair (q,e) and (X2, Y2) for
penalty pair (q2, e2). State variables (signed int8):

| variable | definition |
|----------|-----------|
| `U[r,t]` | `H(i,j) − H(i,j−1)`  (no +p offset) |
| `V[r,t]` | `H(i,j) − H(i−1,j)`  (no +p offset) |
| `X[r,t]` | `F(i,j+1) − H(i,j)` ∈ [−p, 0] when gap active, −p when inactive |
| `Y[r,t]` | `E(i+1,j) − H(i,j)` ∈ [−p, 0] when gap active, −p when inactive |
| `X2[r,t]`| `F2(i,j+1) − H(i,j)` for second pair |
| `Y2[r,t]`| `E2(i+1,j) − H(i,j)` for second pair |

Z intermediate: `Z[r,t] = H(i,j) − H(i−1,j−1)` (no 2p offset — raw diagonal difference).

Update equations:

```
// block1
z   = s(i,j)                              // substitution — no 2p added
a   = X[r-1,t-1] + V[r-1,t-1]           // F-gap path
b   = Y[r-1,t]   + U[r-1,t]             // E-gap path
a2  = X2[r-1,t-1] + V[r-1,t-1]         // F2-gap path
b2  = Y2[r-1,t]   + U[r-1,t]           // E2-gap path

Z   = max(z, a, b, a2, b2)              // signed max (SSE4.1 _mm_max_epi8)
Z   = min(Z, s_match)                   // clamp

// block2
U[r,t]  = Z − V[r-1,t-1]
V[r,t]  = Z − U[r-1,t]

X[r,t]  = max(a − (Z−q),  0) − p       // max(F(i,j+1)−H(i,j), −p)
Y[r,t]  = max(b − (Z−q),  0) − p
X2[r,t] = max(a2 − (Z−q2), 0) − p2
Y2[r,t] = max(b2 − (Z−q2), 0) − p2
```

H reconstruction: `H[t] += v8[t];` (no `− qe` subtraction, unlike extz) confirms
`V_extd[r,t] = H(i,j) − H(i-1,j)` with no offset.

Arithmetic: signed int8. Requires SSE4.1 (`_mm_max_epi8`) for the score-only
path; falls back to an SSE2 emulation with compare + blend. Initialization is
`memset(u, −q−e, …)` (signed negative baseline representing an inactive gap).

## ksw_exts2_sse — splice-aware, signed int8 encoding

**Source**: `ksw2_exts2_sse.c` (RNA-seq / splice mode). **Out of the VecRec
drop-in regime** — left stock.

Carries three gap-state channels: (X, Y) for short-gap pair (q,e) and X2 for
intron jumps. No Y2 — introns only open from the query side. Intron state:

    X2[r,t] = intron_state(i,j+1) − H(i,j)

Intron-open cost is `q2` (a large constant). There is **no per-base extension
penalty** for introns; closing is done via the `acceptor[t]` bonus array (GT-AG,
GC-AG, AT-AC signals). This is a 4-state finite automaton {M, E-gap, F-gap,
intron} where the intron→M transition is position-gated (splice signal) rather
than cost-gated — no affine-gap formulation can express it.

## Recurrence class of each kernel

| kernel | recurrence class |
|--------|-----------------|
| `ksw_extz2_sse` | Gotoh (1-piece affine) |
| `ksw_extd2_sse` | dual-affine (2-piece affine — strict superset of Gotoh) |
| `ksw_exts2_sse` | splice-aware automaton (not affine) |

- **ksw_extz2_sse** is exactly Gotoh given the precondition `−min_score ≤ 2(q+e)`;
  the uint8 clamp is provably non-binding under it.
- **ksw_extd2_sse** carries two parallel gap-state pairs:

      H(i,j) = max( H(i-1,j-1)+s, E(i,j), F(i,j), E2(i,j), F2(i,j) )
      E(i,j)  = max( H(i-1,j) − p,   E(i-1,j) − e  )
      E2(i,j) = max( H(i-1,j) − p2,  E2(i-1,j) − e2 )

  It degenerates to Gotoh when (q,e)=(q2,e2), else approximates a convex gap
  cost; the crossover is `long_thres = (q2−q)/(e−e2) − 1`. The SIMD encoding is
  the same (U,V,X,Y) isomorphism as extz but signed (no +p shift), because the
  second gap pair exhausts uint8 headroom.

## The isomorphism

The mapping `(H, E, F) ↔ (U, V, X, Y)` is a bijection given boundary conditions
— an isomorphism of the max-plus DP (the same argmax path wins in both):

    H(i,j)   = H(0,0) + Σ_{path} V[r',t']
    F(i,j+1) = X[r,t] + H(i,j) − p            (extz encoding)
    E(i+1,j) = Y[r,t] + H(i,j) − p

The two internal encodings (extz unsigned-shifted, extd signed-unshifted) differ
by an affine coordinate change: `U_extz = U_extd + p`, `V_extz = V_extd + p`,
`X_extz = X_extd + p`. The +p shift in extz keeps values ≥ 0 for unsigned
`max_epu8` (SSE2); extd's unshifted signed form needs `max_epi8` (SSE4.1) but
frees the 2p headroom for the second gap pair.

---

# VecRec drop-in

**Files:**
- `generated/vecrec_affine_kernel.h` — generated VecRec kernels (dedicated
  1-piece + 2-piece).
- `generated/ksw2_vecrec_ext.cpp` — wrapper replacing `ksw_extz2_sse` /
  `ksw_extd2_sse` (generated from the `SRC` template in `generate_kernel.py`).

**1-piece vs 2-piece:** `ksw_extz2_sse` (called when `q==q2 && e==e2`) uses a
dedicated 1-piece Gotoh kernel `vecrec_ext_core_1p` (H,E1,F1 — no E2/F2 work).
`ksw_extd2_sse` uses `vecrec_ext_core_2p`. `-DVR_FORCE_2P=1` instead routes the
1-piece entry through the 2-piece core (`q2=q,e2=e` collapse) for an A/B.

**Fields populated** in `ksw_extz_t` (everything minimap2 reads):
- `max`/`max_q`/`max_t` — global DP maximum, not just boundary cells;
- `mqe`/`mqe_t`, `mte`/`mte_q` — separately tracked endpoint stats;
- `reach_end` — set when `mqe + end_bonus > global_max` (extension mode only);
- `cigar`/`n_cigar` — real traceback CIGAR with exact direction bits;
- `zdropped = 0` always.

**reach_end + traceback-start** are one coupled decision (mirrors
`ksw2_extd2_sse.c`) so they can't diverge: gap-fill → corner `(qlen,tlen)`,
`reach_end=0`; `EXTZ_ONLY` and `mqe+end_bonus > gmax` → start `(qlen, mqe_t)`,
`reach_end=1`; else → `(gmax_i, gmax_j)`, `reach_end=0`. Two traceback routines
selected by `KSW_EZ_RIGHT` (gap-LEFT default; gap-RIGHT on left-extension).

**`ez->score` vs `ez->max`:** extensions read `ez->max` (=gmax); gap-fill reads
`ez->score` (must be the corner `H(qlen,tlen)`). Get this wrong and DP scores
drift silently.

**Memory:** compiled `-DHAVE_KALLOC` so all scratch (`profile`, `sym`, `Hmat`,
`ops`, CIGAR) uses minimap2's kalloc pool, not system malloc.

**Dual element width (I8 / I16).** `generate_kernel.py` emits both an I8 and an
I16 kernel per piece-count (`k_vr_{1p,2p}_{i8,i16}`). The entry points dispatch
on an upper bound of the achievable score: when
`min(qlen,tlen)*max_match + end_bonus <= VR_I8_MAX` (default `120`, see
[flags.py](flags.py)) the call uses the I8 kernel, otherwise the I16 kernel.

Saturating I8/I16 kernels use an internal "-infinity" sentinel as the identity
of their `max()` reductions (`vec-rec/src/vecrec/util.py`'s `val_to_str` and the
`TropMaxSat`/`TropMinSat` instantiations in `vec-rec/src/vecrec/templates/common.h`).
A real score more negative than that sentinel would be silently wrong. As a
guard, `VR_I8_SAFE_MAG` / `VR_I16_SAFE_MAG` in `generate_kernel.py` bound the
worst-case score magnitude of a call (the DP corner is always reachable via
`min(qlen,tlen)` diagonal steps + `|qlen-tlen|` gap steps, so the optimum is
never worse than that path) and delegate the call to stock ksw2 when it could
reach either width's limit. These thresholds must stay below the sentinel
magnitude that the pinned vec-rec actually emits.

To check a width directly, instantiate `k_vr_2p_i8` (or the `1p` / `i16`
variants) from `generated/vecrec_affine_kernel.h` with `qlen=0`, `tlen>=100`
and asm20 gap parameters (`q=6,e=2,q2=26,e2=1`), drive it exactly like the
`CORE_TEMPLATE` loop (take EPI/vector lanes from the kernel's own
`*_vector_type`; multi-pipelined schedules process more than one `vec_type`
width per `run()` call), and compare the H(0,j) row against a scalar
`max(Ht_left+go, F+ge)` reference. Inside minimap2, `VR_CHECK=2` (below) checks
every cell on real calls.

**Banded mode.** VecRec tracks the DP max inline, so it has no separate
approximate-max mode. With a band (`w>0`), `seek_col` skips out-of-band columns.
The `approx-banded*` scenarios run stock minimap2 with approx-max + its default
band against the VecRec drop-in in banded mode; the `exact-*` scenarios use the
fair regime above.

**Known limitation — no z-drop:** extension calls are correct without it (full
band computed, true optimum returned). Gap-fill calls without z-drop won't split
chimeras/inversions into separate mappings. Hence `-z 1000000` is mandatory.

---

# Ablation & instrumentation reference

## Compile-time flags (VecRec core)

Pass via `VECREC_EXTRA_DEFS` (forwarded by `build_minimap2.sh`). Compile-time, so
**each variant is a separate build**.

| Flag | Effect | Correct? |
|------|--------|----------|
| `VR_FORCE_2P=1` | Route the 1-piece entry through the 2-piece core (`q2=q,e2=e` collapse), for an A/B against the dedicated 1-piece kernel. | yes |
| `VR_NO_TRACEBACK=1` | Skip traceback; emit trivial CIGAR. Isolates traceback cost. | no (timing only) |
| `VR_DUMMY_PROFILE=1` | Constant score profile, skips the O(m·tlen) build. Isolates profile-build cost. | no (timing only) |
| `VR_NO_HSTORE=1` | Skip the per-chunk `memcpy` into `Hmat`. Isolates H store traffic. Implies `VR_NO_TRACEBACK`. | no (timing only) |
| `VR_NO_ENDPOINT_SCAN=1` | Skip mqe/mte last-row / last-column scans. | no (timing only) |
| `VR_CHECK=1` | After each call, recompute with an independent scalar int32 reference DP; assert corner + global max match. | yes (gate) |
| `VR_CHECK=2` | As `=1`, plus compare every `Hmat` cell to a full int32 reference matrix. Strongest; O(N) extra memory. | yes (gate) |
| `VR_I8_MAX=N` | Max-score threshold for the I8/I16 dispatch: a call uses the I8 kernel when `min(qlen,tlen)*max_match + end_bonus <= N` (default `120`; `0` disables I8). See "Dual element width" above. | yes |

```bash
VECREC_EXTRA_DEFS="-DVR_NO_TRACEBACK=1 -DVR_DUMMY_PROFILE=1" \
  bash end_to_end_baselines/scripts/build_minimap2.sh
```

Timing-only ablations emit garbage alignments that trip minimap2's CIGAR-span
asserts, aborting before reports print. `build_minimap2.sh` therefore auto-defines
`VR_UNSAFE` (disabling only those two output-consistency asserts in `align.c`)
whenever a timing-only `VR_*` flag is present; force with `VECREC_UNSAFE=1`. The
regime asserts (banding / score-only / z-drop) stay active.

**How VR_CHECK works:** an independent scalar int32 full-matrix DP (Gotoh for
1-piece, dual-affine for 2-piece — rolling two rows, no SIMD, no difference
encoding) runs alongside the VecRec DP on every real call and `assert(0)`s on
mismatch, dumping qlen/tlen/expected/got. It catches encoding/codegen bugs the
SIMD path could share. Correctness gate only; do not use it for timing runs.

## Compile flags shared by both binaries (align.c / ksw2)

Carried via `MM_DEFS` (into the C build of BOTH binaries). Declared in
[flags.py](flags.py); `run.py` routes a scenario's `defines` into the right build
env var automatically, so adding a flag = one `flags.py` entry + one source `#ifdef`.

| Flag | Default | Effect |
|------|---------|--------|
| `MM_EXACT_ONLY` | `1` (exact) | Bake exact-max on gap-fill pass-1 (ignore `KSW_EZ_APPROX_MAX`) — the fair regime. Build `-DMM_EXACT_ONLY=0` to restore stock's approx-max shortcut for an approx-vs-exact A/B. |
| `MM_COUNT_MATRIX` | off | Print one `[extend-size]` line at exit (per-call qlen/tlen/cells + max(q,t) histogram), **no** per-call clock overhead. |
| `VECREC_PHASE_TIMING` | off | Per-call setup/dp/tb timers + `[extend-*]` / `[vecrec-phases]` reports. Shared counters contend under `-t N`; **single-thread profiling only**. |

```bash
MM_DEFS="-DMM_COUNT_MATRIX"  bash end_to_end_baselines/scripts/build_minimap2.sh
MM_DEFS="-DMM_EXACT_ONLY=0"  bash end_to_end_baselines/scripts/build_minimap2.sh  # approx A/B
```

## Other build-level env

| Env | Effect |
|-----|--------|
| `DEBUG=1` | `-O0 -g` binaries with a `_debug` suffix. |
| `VECREC_MAX_BITS=256\|512` | Vector width (epi≈32 vs 64). |
| `ZLIB_DIR=<dir>` | Local zlib for hosts without dev headers. |
| `VECREC_EXTRA_DEFS="-DVR_*"` | VecRec-wrapper ablation switches (above). |
| `VECREC_UNSAFE=1` | Disable minimap2's CIGAR-span asserts (auto-set for timing-only `VR_*`). |

> The approx-max policy is now a **compile-time** flag (`MM_EXACT_ONLY`, default
> on = exact). The old runtime `MM_NO_APPROX` / `MM_USE_APPROX` env knobs are retired.

## Instrumentation reports (with `VECREC_PHASE_TIMING=1`, single-thread)

Both binaries print to stderr at process exit (`align.c` is compiled into both):

- `[extend-counts]` — total / 1-piece / 2-piece / splice / approx-requested /
  approx-delegated / left / right / gapfill call counts.
- `[extend-timing]` — mean / p50 / p95 / p99 µs/call, total extend seconds, %-of-runtime.
- `[extend-size]` — mean qlen, mean tlen, mean cells = mean(qlen·tlen), max,
  full-matrix fraction, plus an exact-only histogram bucketed on max(qlen,tlen).
- `[extend-phases]` (stock) / `[vecrec-phases]` (vecrec) — setup / DP / traceback
  split per call, so the two builds line up phase-by-phase.

VecRec-side phase markers live in the `SRC` template in `generate_kernel.py`
(not the generated file — it is overwritten every build): `vr_init_ns` (alloc +
ctor), `vr_dp_ns` (sweep + mqe/mte scan), `vr_tb_ns` (traceback + CIGAR). Stock
markers live in `ksw2_extd2_sse.c` → `mm_ksw_setup_ns` / `mm_ksw_dp_ns` /
`mm_ksw_tb_ns`, read by a `__attribute__((destructor))` in `align.c`. Always use
`__atomic_fetch_add(..., __ATOMIC_RELAXED)` so `-t N` stays correct.

## Always-on assertions (both builds, not gated by timing)

Enforce the drop-in regime so no run silently leaves it (no `-DNDEBUG`):

- not score-only (`!(flag & KSW_EZ_SCORE_ONLY)`);
- band inactive (`w <= 0 || w >= max(qlen,tlen)-1`), plus a per-anti-diagonal
  no-clip check inside the stock kernels;
- CIGAR wanted (`with_cigar`);
- z-drop param huge (`zdrop <= 0 || zdrop >= 100000`) and never fires
  (`!ez->zdropped`, plus `assert(0)` at both `ksw_apply_zdrop` sites);
- splice path out of regime (`assert(0)` — do not run `-ax splice`).

Any assertion aborts with a message; hitting one means the run is out of the
supported regime (usually a forgotten `-z 1000000` or a divergent/large input
that triggers real banding).

---

# Benchmark recipes

Prefer the scenario runner (`run.py`) — it builds with the right defines and runs
parity + timing. The recipes below are the low-level equivalents. Default
reference/reads are `end_to_end_baselines/data/MT-human.fa` and
`end_to_end_baselines/data/mt_reads.fq`. Always pass `-z 1000000`.

## Correctness / parity (run first)

```bash
uv run python end_to_end_baselines/minimap2/run.py exact-smoke                  # via runner
uv run python end_to_end_baselines/scripts/compare_e2e_sam.py --zdrop 1000000   # low-level
# expected: same RNAME:POS, same AS score, ~same CIGAR (H-only traceback may
# differ at equal score)
```

## Phase profile (profiling build only)

```bash
VECREC_PHASE_TIMING=1 bash end_to_end_baselines/scripts/build_minimap2.sh
MT=end_to_end_baselines/data/MT-human.fa
READS=end_to_end_baselines/data/mt_reads.fq
FLAGS="-ax map-ont -t 1 --secondary=no -z 1000000"

bin/minimap2_original $FLAGS $MT $READS 2>&1 >/dev/null | grep '\[extend-'    # exact by default
bin/minimap2_vecrec   $FLAGS $MT $READS 2>&1 >/dev/null | grep -E '\[extend-|\[vecrec-'
# then rebuild WITHOUT timing before any perf/thread run.
```

## Ablation sweep

```bash
uv run python end_to_end_baselines/scripts/run_ablation.py --reps 3
# add --extra-opts "-O4,4 -E2,2" to force the 1-piece kernel path
```

Rebuilds one variant per row (compile-time flags), times each with phase timers
on, prints µs/call (mean/p50/p95), the init/dp/tb split, and per-phase delta vs
the full kernel.

## Correctness gate on real calls

```bash
VECREC_EXTRA_DEFS="-DVR_CHECK=2" bash end_to_end_baselines/scripts/build_minimap2.sh
bin/minimap2_vecrec -ax map-ont -t1 --secondary=no -z 1000000 $MT $READS >/dev/null
```

## Thread sweep (use timing-OFF binaries)

`VECREC_PHASE_TIMING` builds share atomic counters across threads (see above),
so multi-threaded timing runs must use binaries built with timing OFF (the
default). `mt_reads_e3` (1000 reads) is a larger alternative to `mt_reads` (200).

## Scope check

Divergent / large-genome inputs activate a real band; the full-matrix drop-in
asserts and aborts rather than mis-aligning. Probe:

```bash
scope(){ bin/minimap2_vecrec -ax map-ont -t4 --secondary=no -z 1000000 "$1" "$2" >/dev/null 2>/tmp/s.err;
         grep -q Assertion /tmp/s.err && echo "OUT-OF-SCOPE (banded)" || echo "in-scope"; }
```

## Gotchas

- **Regenerated file**: `generated/ksw2_vecrec_ext.cpp` is produced by
  `generate_kernel.py` on every build — edit the `SRC` template, never the output.
- **VecRec needs clang**: the wrapper `.o` is compiled with `clang++`; C sources use `cc`.
- **z-drop must be off**: always pass `-z 1000000`.
- **Non-banding is enforced**: a genuinely-banded affine call aborts loudly.
- **`ez->score` vs `ez->max`**: extensions use `ez->max`; gap-fill uses `ez->score` (corner).
- **Isolated kernel timing** is obtained via compile flags on the e2e binary
  (`VR_NO_TRACEBACK`, `VR_DUMMY_PROFILE`, `VECREC_PHASE_TIMING`), not a separate
  microbench.
