"""Scenario registry for the minimap2 VecRec comparison.

A Scenario ties together: a dataset (datasets.py), the minimap2 runtime knobs
(preset / bandwidth -r / z-drop -z), the compile-time build defines (flags.py),
which metrics to collect, and which binaries to compare. run.py builds the
needed binary variant (keyed by define-set) and runs the comparison.

Extending the harness = add a Scenario entry here. Nothing in run.py changes.

The seeded scenarios cover Scenario 1 (exact: VecRec exact vs stock exact, no
band / no z-drop / no approx — VecRec's fair regime). `approx-banded` is a stub
wired for the later Scenario 2 work (VecRec's own approx-max + banded path vs
stock approx + default banding).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# large -r so the diagonal band never clips (fair full-matrix regime)
WIDE_BW = 100_000
# large -z so z-drop never fires
NO_ZDROP = 1_000_000


@dataclass(frozen=True)
class Scenario:
    name: str
    dataset: str                 # key into datasets.DATASETS
    preset: str = "map-ont"      # minimap2 -x preset
    bw: int | None = WIDE_BW     # -r NUM,NUM; None -> minimap2 default (banded)
    zdrop: int | None = NO_ZDROP  # -z NUM,NUM; None -> minimap2 preset default (real z-drop, can fire)
    defines: list[str] = field(default_factory=list)  # flags.py define tokens
    metrics: tuple[str, ...] = ("parity", "timing")   # subset of {parity, timing}
    stub: bool = False           # not yet runnable (placeholder for future work)
    desc: str = ""


SCENARIOS: dict[str, Scenario] = {
    # ── Scenario 1: exact vs exact (fair regime) ──────────────────────────────
    "exact-smoke": Scenario(
        name="exact-smoke",
        dataset="parity-smoke",
        metrics=("parity",),
        desc="Fast correctness/parity gate on MT-human (tiny DP).",
    ),
    "exact-sv": Scenario(
        name="exact-sv",
        dataset="sv-fair",
        metrics=("parity", "timing"),
        desc="Primary fair perf: large-extension SV reads, exact vs exact, "
             "no band / no z-drop / no approx.",
    ),
    "exact-tandem": Scenario(
        name="exact-tandem",
        dataset="tandem",
        metrics=("timing",),
        desc="Repeat-heavy 15kbp extensions, exact vs exact.",
    ),
    "exact-crossspecies": Scenario(
        name="exact-crossspecies",
        dataset="crossspecies-real",
        preset="asm20",
        metrics=("timing",),
        desc="Real cross-species (E. coli vs Salmonella), asm20, exact vs exact.",
    ),
    "exact-ecoli-ultralong": Scenario(
        name="exact-ecoli-ultralong",
        dataset="ecoli-ultralong-real",
        metrics=("parity", "timing"),
        desc="Real ultra-long ONT E. coli reads vs real E. coli K-12 reference, "
             "exact vs exact.",
    ),
    # ── Scenario 1b: exact (VecRec) vs approx-max only (stock), no banding ────
    # Isolates the approx-max shortcut alone, decoupled from banding (band
    # stays wide, i.e. off). Gap-fill calls never set KSW_EZ_EXTZ_ONLY, so in
    # both ksw2_ext{z,d}2_sse.c the with_cigar traceback always takes the
    # corner path `(tlen-1, qlen-1)` for gap-fill regardless of approx_max --
    # only the exact 32-bit H/max bookkeeping (lines computing ez->max/max_t,
    # skipped when approx_max) is skipped. With z-drop disabled (required by
    # the regime) that bookkeeping never changes the result, so parity should
    # hold exactly here; the difference is pure stock-side speed from
    # skipping that bookkeeping, with no confound from banding (contrast with
    # approx-banded below, which also changes VecRec's own traceback).
    "approx-max": Scenario(
        name="approx-max",
        dataset="sv-fair",
        defines=["MM_EXACT_ONLY=0"],  # stock uses its approx-max shortcut (gap-fill p1)
        metrics=("parity", "timing"),
        desc="VecRec exact-max (always) vs stock approx-max shortcut, isolated "
             "from banding (band stays wide). Large-extension SV reads. Parity "
             "expected to hold (approx-max only skips global-max bookkeeping, "
             "not the corner traceback gap-fill actually uses); timing shows "
             "the shortcut's speed cost on its own, without the band confound.",
    ),
    # ── Scenario 2: banded fast path vs stock approx + default banding ────────
    # VecRec is already row-major and tracks the DP max inline exactly, so it has
    # no separate lossy approx-max kernel; its "fast" path is banded static
    # pruning (seek_col skips out-of-band columns). This compares that against
    # stock ksw2's approx-max shortcut under minimap2's default banding, on long
    # reads where the band actually clips (short reads make w >= len => no band).
    "approx-banded": Scenario(
        name="approx-banded",
        dataset="sv-fair",            # large-extension DPs (~2500^2) that the default band clips
        bw=None,                      # minimap2 default banding (no wide -r)
        defines=["MM_EXACT_ONLY=0"],  # stock uses its approx-max shortcut (gap-fill p1)
        metrics=("timing",),
        desc="VecRec banded static-pruning (band-packed H; row-major, exact-max "
             "inline) vs stock approx-max + default banding, on large-extension SV "
             "reads where the band actually clips. Timing only: VecRec's banded "
             "traceback is approximate at band edges, so per-read parity is not "
             "expected here (unlike the exact scenarios).",
    ),
    "approx-banded-tandem": Scenario(
        name="approx-banded-tandem",
        dataset="tandem",              # 15kbp repeat-heavy extensions, default band clips
        bw=None,
        defines=["MM_EXACT_ONLY=0"],
        metrics=("timing",),
        desc="Same as approx-banded (stock approx-max + default band vs VecRec "
             "banded static-pruning), on the repeat-heavy tandem dataset instead "
             "of sv-fair.",
    ),
    "approx-banded-crossspecies": Scenario(
        name="approx-banded-crossspecies",
        dataset="crossspecies-real",   # real E. coli vs Salmonella, default band clips
        preset="asm20",
        bw=None,
        defines=["MM_EXACT_ONLY=0"],
        metrics=("timing",),
        desc="Same as approx-banded, on the real cross-species (asm20) dataset "
             "instead of sv-fair.",
    ),
    "approx-banded-ecoli-ultralong": Scenario(
        name="approx-banded-ecoli-ultralong",
        dataset="ecoli-ultralong-real",
        bw=None,
        defines=["MM_EXACT_ONLY=0"],
        metrics=("timing",),
        desc="Same as approx-banded, on real ultra-long ONT E. coli reads vs "
             "the real E. coli K-12 reference.",
    ),
    # ── 3-tier real-data ablation: no-approx | static-approx | dynamic-approx ─
    # Same dataset, three regimes of increasing "shortcut-taking" by stock:
    #   no-approx:      no band, no approx-max, no z-drop (both binaries exact)
    #   static-approx:  real default band + stock approx-max, z-drop still OFF
    #                    (VecRec's own shortcut is its banded static-pruning path)
    #   dynamic-approx: static-approx + real default z-drop actually ON (this
    #                    requires the z-drop `assert(0)`s removed from
    #                    ksw2_extd2_sse.c -- done; VecRec has no z-drop mechanism
    #                    so it behaves identically to static-approx here, while
    #                    stock gets to actually exploit the early-exit).
    # These no longer force VR_I8_MAX=0: the I8/I16 sentinel bug that made I8
    # unsafe (see flags.py's VR_I8_MAX entry / CLAUDE.md's "Dual element
    # width" section) was root-caused and fixed upstream in vec-rec; I8 is
    # back at its default (120) and genuinely correct. NOTE: the "no-approx"
    # tier's wide `-r 100000,100000` can still push a call's *worst-case*
    # achievable magnitude (see VR_I8_SAFE_MAG/VR_I16_SAFE_MAG in
    # generate_kernel.py) past what even the widened I16 sentinel can safely
    # represent -- when that happens VecRec's own wrapper delegates that call
    # to stock ksw2. This is correct behavior (not a bug), but it means
    # datasets with large, genuinely divergent gap-fills under the unbanded
    # "no-approx" regime (real cross-species / same-species strain
    # comparisons especially) can see the no-approx speedup drop or invert,
    # because VecRec is silently running stock's code for its biggest,
    # would-have-won-most calls. Numbers in this file's history / the
    # run-minimap2-approx-ablation skill may be stale if vec-rec's kernel has
    # changed since -- always treat a fresh `--rebuild` run as authoritative.
    "ablation-noapprox-ecoli-ultralong": Scenario(
        name="ablation-noapprox-ecoli-ultralong",
        dataset="ecoli-ultralong-real",
        defines=[],
        metrics=("parity", "timing"),
        desc="Tier 1/3: no band, no approx-max, no z-drop. Both binaries exact.",
    ),
    "ablation-staticapprox-ecoli-ultralong": Scenario(
        name="ablation-staticapprox-ecoli-ultralong",
        dataset="ecoli-ultralong-real",
        bw=None,                       # real default band (map-ont preset)
        defines=["MM_EXACT_ONLY=0"],
        metrics=("timing",),
        desc="Tier 2/3: real default band + stock approx-max; z-drop still OFF. "
             "VecRec's banded static-pruning path vs stock's approx-max+band.",
    ),
    "ablation-dynamicapprox-ecoli-ultralong": Scenario(
        name="ablation-dynamicapprox-ecoli-ultralong",
        dataset="ecoli-ultralong-real",
        bw=None,                       # real default band
        zdrop=None,                    # real default z-drop (map-ont: 400/200) -- CAN fire
        defines=["MM_EXACT_ONLY=0"],
        metrics=("timing",),
        desc="Tier 3/3: real default band + approx-max + real default z-drop "
             "(actually allowed to fire for stock). VecRec has no z-drop "
             "mechanism so it behaves like static-approx here -- this measures "
             "the cost of that missing capability against stock's full "
             "real-world shortcut set.",
    ),
    "ablation-noapprox-ecoli-strains": Scenario(
        name="ablation-noapprox-ecoli-strains",
        dataset="ecoli-strains-real",
        preset="asm10",                # ~1-2% intra-species core divergence
        defines=[],
        metrics=("timing",),           # parity checker is map-ont-tuned; skip
        desc="Tier 1/3 on real same-species E. coli strains (K-12 vs O157:H7, "
             "asm10).",
    ),
    "ablation-staticapprox-ecoli-strains": Scenario(
        name="ablation-staticapprox-ecoli-strains",
        dataset="ecoli-strains-real",
        preset="asm10",
        bw=None,                       # real default band (asm* preset: bw=1000)
        defines=["MM_EXACT_ONLY=0"],
        metrics=("timing",),
        desc="Tier 2/3 on real same-species E. coli strains: real default "
             "band + stock approx-max, z-drop still off.",
    ),
    "ablation-dynamicapprox-ecoli-strains": Scenario(
        name="ablation-dynamicapprox-ecoli-strains",
        dataset="ecoli-strains-real",
        preset="asm10",
        bw=None,
        zdrop=None,                    # real default z-drop (asm* preset: 200) -- CAN fire
        defines=["MM_EXACT_ONLY=0"],
        metrics=("timing",),
        desc="Tier 3/3 on real same-species E. coli strains: real default "
             "band + approx-max + real default z-drop.",
    ),
    # ── Same strains data, but map-ont preset instead of asm10 ────────────────
    # asm5/asm10/asm20 all set a real default bw_long=100000 (see options.c) --
    # essentially the same magnitude as this harness's own "no approx" WIDE_BW
    # override, so static-approx never meaningfully restricts long-range joins
    # for those presets (see ablation-staticapprox-ecoli-strains's fallback
    # mismatch). map-ont's real bw_long stays at the global default of 20000 --
    # 5x tighter -- while the same real, substantial (~1-2% + genomic islands)
    # divergence should still produce large-enough gap-fill DP for VecRec's
    # per-cell throughput to matter. Testing whether that combination (real
    # divergence + a preset whose real defaults are genuinely tight) is where
    # static-approx can show a real win.
    "ablation-noapprox-ecoli-strains-mapont": Scenario(
        name="ablation-noapprox-ecoli-strains-mapont",
        dataset="ecoli-strains-real",
        preset="map-ont",
        defines=[],
        metrics=("timing",),
        desc="Tier 1/3 on real same-species E. coli strains, map-ont preset "
             "instead of asm10 (tighter real bw_long=20000 vs asm's 100000).",
    ),
    "ablation-staticapprox-ecoli-strains-mapont": Scenario(
        name="ablation-staticapprox-ecoli-strains-mapont",
        dataset="ecoli-strains-real",
        preset="map-ont",
        bw=None,                       # real default band (map-ont: bw=500, bw_long=20000)
        defines=["MM_EXACT_ONLY=0"],
        metrics=("timing",),
        desc="Tier 2/3 on real same-species E. coli strains, map-ont preset: "
             "real default band + stock approx-max, z-drop still off.",
    ),
    "ablation-dynamicapprox-ecoli-strains-mapont": Scenario(
        name="ablation-dynamicapprox-ecoli-strains-mapont",
        dataset="ecoli-strains-real",
        preset="map-ont",
        bw=None,
        zdrop=None,                    # real default z-drop (map-ont: 400/200) -- CAN fire
        defines=["MM_EXACT_ONLY=0"],
        metrics=("timing",),
        desc="Tier 3/3 on real same-species E. coli strains, map-ont preset.",
    ),
    "ablation-noapprox-mtb-strains": Scenario(
        name="ablation-noapprox-mtb-strains",
        dataset="mtb-strains-real",
        preset="map-ont",
        defines=[],
        metrics=("timing",),
        desc="Tier 1/3 on real M. tuberculosis strains (H37Rv vs CDC1551), "
             "map-ont preset.",
    ),
    "ablation-staticapprox-mtb-strains": Scenario(
        name="ablation-staticapprox-mtb-strains",
        dataset="mtb-strains-real",
        preset="map-ont",
        bw=None,
        defines=["MM_EXACT_ONLY=0"],
        metrics=("timing",),
        desc="Tier 2/3 on real M. tuberculosis strains: real default band + "
             "stock approx-max, z-drop still off. Low genomic plasticity "
             "should keep this collinear (unlike E. coli's genomic islands).",
    ),
    "ablation-dynamicapprox-mtb-strains": Scenario(
        name="ablation-dynamicapprox-mtb-strains",
        dataset="mtb-strains-real",
        preset="map-ont",
        bw=None,
        zdrop=None,
        defines=["MM_EXACT_ONLY=0"],
        metrics=("timing",),
        desc="Tier 3/3 on real M. tuberculosis strains: real default band + "
             "approx-max + real default z-drop.",
    ),
    "ablation-noapprox-ecoli-hifi": Scenario(
        name="ablation-noapprox-ecoli-hifi",
        dataset="ecoli-hifi-real",
        preset="map-hifi",
        defines=[],
        metrics=("parity", "timing"),
        desc="Tier 1/3 on real PacBio HiFi E. coli reads (map-hifi preset).",
    ),
    "ablation-staticapprox-ecoli-hifi": Scenario(
        name="ablation-staticapprox-ecoli-hifi",
        dataset="ecoli-hifi-real",
        preset="map-hifi",
        bw=None,                       # real default band (map-hifi preset: bw=500)
        defines=["MM_EXACT_ONLY=0"],
        metrics=("timing",),
        desc="Tier 2/3 on real HiFi reads: real default band + stock "
             "approx-max, z-drop still off. Low HiFi error rate should keep "
             "the band from clipping much, unlike the noisy ONT set.",
    ),
    "ablation-dynamicapprox-ecoli-hifi": Scenario(
        name="ablation-dynamicapprox-ecoli-hifi",
        dataset="ecoli-hifi-real",
        preset="map-hifi",
        bw=None,
        zdrop=None,                    # real default z-drop -- CAN fire
        defines=["MM_EXACT_ONLY=0"],
        metrics=("timing",),
        desc="Tier 3/3 on real HiFi reads: real default band + approx-max + "
             "real default z-drop. HiFi's high accuracy means few structural "
             "surprises, so z-drop should rarely fire -- the case most "
             "favorable to VecRec's lack of a z-drop mechanism.",
    ),
    "ablation-noapprox-crossspecies": Scenario(
        name="ablation-noapprox-crossspecies",
        dataset="crossspecies-real",
        preset="asm20",
        defines=[],
        metrics=("timing",),
        desc="Tier 1/3 on real cross-species (E. coli vs Salmonella), asm20. "
             "Parity skipped (compare_e2e_sam.py supports map-ont only).",
    ),
    "ablation-staticapprox-crossspecies": Scenario(
        name="ablation-staticapprox-crossspecies",
        dataset="crossspecies-real",
        preset="asm20",
        bw=None,
        defines=["MM_EXACT_ONLY=0"],
        metrics=("timing",),
        desc="Tier 2/3 on real cross-species, asm20 real default band.",
    ),
    "ablation-dynamicapprox-crossspecies": Scenario(
        name="ablation-dynamicapprox-crossspecies",
        dataset="crossspecies-real",
        preset="asm20",
        bw=None,
        zdrop=None,                    # asm20 real default z-drop (200) -- CAN fire
        defines=["MM_EXACT_ONLY=0"],
        metrics=("timing",),
        desc="Tier 3/3 on real cross-species, asm20 real default band + z-drop.",
    ),
    # ── New map-ont candidates: exact-DP win + approx fidelity cost ───────────
    # Selected to clone the two wins (ecoli-ultralong / ecoli-strains map-ont):
    # same-species or same-organism, map-ont preset, real error or ~1-2% strain
    # divergence with islands. Avoid HiFi / MTB-like clonal / cross-species asm*.
    "ablation-noapprox-ecoli-ultralong-full": Scenario(
        name="ablation-noapprox-ecoli-ultralong-full",
        dataset="ecoli-ultralong-full",
        defines=[],
        metrics=("timing",),
        desc="Tier 1/3 on 500 longest full-run ultra-long ONT E. coli reads "
             "(no band, exact-max, no z-drop).",
    ),
    "ablation-staticapprox-ecoli-ultralong-full": Scenario(
        name="ablation-staticapprox-ecoli-ultralong-full",
        dataset="ecoli-ultralong-full",
        bw=None,
        defines=["MM_EXACT_ONLY=0"],
        metrics=("timing",),
        desc="Tier 2/3 on full-run ultra-long ONT subsample.",
    ),
    "ablation-dynamicapprox-ecoli-ultralong-full": Scenario(
        name="ablation-dynamicapprox-ecoli-ultralong-full",
        dataset="ecoli-ultralong-full",
        bw=None,
        zdrop=None,
        defines=["MM_EXACT_ONLY=0"],
        metrics=("timing",),
        desc="Tier 3/3 on full-run ultra-long ONT subsample.",
    ),
    "ablation-noapprox-pa-strains": Scenario(
        name="ablation-noapprox-pa-strains",
        dataset="pa-strains-real",
        preset="map-ont",
        defines=[],
        metrics=("timing",),
        desc="Tier 1/3 on P. aeruginosa PA14 vs PAO1 (map-ont): "
             "no band, exact-max, no z-drop.",
    ),
    "ablation-staticapprox-pa-strains": Scenario(
        name="ablation-staticapprox-pa-strains",
        dataset="pa-strains-real",
        preset="map-ont",
        bw=None,
        defines=["MM_EXACT_ONLY=0"],
        metrics=("timing",),
        desc="Tier 2/3 on P. aeruginosa PA14 vs PAO1 (map-ont).",
    ),
    "ablation-dynamicapprox-pa-strains": Scenario(
        name="ablation-dynamicapprox-pa-strains",
        dataset="pa-strains-real",
        preset="map-ont",
        bw=None,
        zdrop=None,
        defines=["MM_EXACT_ONLY=0"],
        metrics=("timing",),
        desc="Tier 3/3 on P. aeruginosa PA14 vs PAO1 (map-ont).",
    ),
    "ablation-noapprox-salm-strains": Scenario(
        name="ablation-noapprox-salm-strains",
        dataset="salm-strains-real",
        preset="map-ont",
        defines=[],
        metrics=("timing",),
        desc="Tier 1/3 on S. Typhimurium SL1344 vs LT2 (map-ont): "
             "no band, exact-max, no z-drop.",
    ),
    "ablation-staticapprox-salm-strains": Scenario(
        name="ablation-staticapprox-salm-strains",
        dataset="salm-strains-real",
        preset="map-ont",
        bw=None,
        defines=["MM_EXACT_ONLY=0"],
        metrics=("timing",),
        desc="Tier 2/3 on S. Typhimurium SL1344 vs LT2 (map-ont).",
    ),
    "ablation-dynamicapprox-salm-strains": Scenario(
        name="ablation-dynamicapprox-salm-strains",
        dataset="salm-strains-real",
        preset="map-ont",
        bw=None,
        zdrop=None,
        defines=["MM_EXACT_ONLY=0"],
        metrics=("timing",),
        desc="Tier 3/3 on S. Typhimurium SL1344 vs LT2 (map-ont).",
    ),
    # ── Scenario 3: force I8 on every call, no matter the size ────────────────
    # VR_I8_MAX gates the per-call I8-vs-I16 dispatch in generate_kernel.py:
    # `_ub = min(qlen,tlen)*max_match + end_bonus <= VR_I8_MAX` picks the narrow
    # (2x-lane) I8 core, else the always-correct I16 core. Cranking the
    # threshold arbitrarily high forces I8 on every call regardless of size --
    # deliberately unsafe past whatever N this scoring scheme saturates I8 at.
    # Use sweep_sizes.py with --parity to find where correctness breaks and
    # what the speedup looks like right up to that point.
    "i8-forced": Scenario(
        name="i8-forced",
        dataset="sv-fair",
        defines=["VR_I8_MAX=999999999"],  # force I8 unconditionally (see above)
        metrics=("parity", "timing"),
        desc="Force VecRec's I8 core on every call (ignore the safe dispatch "
             "threshold). Exact-vs-exact bw/zdrop regime otherwise (isolates "
             "the width-forcing effect alone). Expect PASS at small N, then a "
             "correctness FAIL once scores exceed int8's saturating range --"
             "use sweep_sizes.py --parity to find that N and the speedup up to it.",
    ),
}


def get(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; known: {', '.join(SCENARIOS)}")
    return SCENARIOS[name]
