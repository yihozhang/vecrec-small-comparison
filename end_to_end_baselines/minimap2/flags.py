"""Compile-flag registry for the minimap2 VecRec comparison harness.

This is the single source of truth for the build-time `-D` flags that shape a
minimap2 build. Each `Flag` declares which build env var carries it (its
"bucket") and which binaries it affects. `run.py` uses this to (a) route a
scenario's requested flags into the right env var for `build_minimap2.sh`, and
(b) key the built-binary cache on the full define-set so scenarios that share a
define-set reuse one binary.

Extending: adding a flag = add one `Flag` entry here + the matching `#ifdef` in
the source (align.c / ksw2_*.c for the C build, ksw2_vecrec_ext.cpp for the
wrapper). Nothing in `build_minimap2.sh` or `run.py` needs to change.

Buckets -> the build_minimap2.sh env var that carries the flag:
  BOTH   -> MM_DEFS            : compiled into align.c/ksw2 (both binaries)
  VECREC -> VECREC_EXTRA_DEFS  : compiled into the vecrec wrapper only
  TIMING -> VECREC_PHASE_TIMING: dedicated toggle (spans C + wrapper + orig objs)
"""
from __future__ import annotations

from dataclasses import dataclass

BOTH = "both"
VECREC = "vecrec"
TIMING = "timing"


@dataclass(frozen=True)
class Flag:
    name: str          # the -D token name, e.g. "MM_EXACT_ONLY"
    bucket: str         # BOTH | VECREC | TIMING
    default: str        # human description of the default
    applies_to: str     # stock | vecrec | both (which binary the effect shows in)
    desc: str


FLAGS: list[Flag] = [
    Flag("MM_EXACT_ONLY", BOTH, "1 (exact max; approx off)", "both",
         "Bake exact-max on gap-fill pass-1 (ignore KSW_EZ_APPROX_MAX). Default "
         "1 = the fair kernel-vs-kernel regime. Set -DMM_EXACT_ONLY=0 to restore "
         "stock's approx-max shortcut for an approx-vs-exact A/B."),
    Flag("MM_COUNT_MATRIX", BOTH, "off", "both",
         "Print one [extend-size] line at exit (per-call qlen/tlen/cells + a "
         "max(q,t) histogram) with no per-call clock overhead."),
    Flag("VECREC_PHASE_TIMING", TIMING, "off", "both",
         "Per-call setup/dp/tb timers + [extend-*] / [vecrec-phases] reports. "
         "Single-thread profiling only (shared counters throttle -t N)."),
    Flag("VR_CHECK", VECREC, "off", "vecrec",
         "Scalar-reference correctness verify per call (=1 corner+global max, "
         "=2 every Hmat cell). Correctness gate, never a perf run."),
    Flag("VR_FORCE_2P", VECREC, "off", "vecrec",
         "Route the 1-piece entry (ksw_extz2_sse) through the 2-piece core; "
         "A/B the dedicated-1-piece win."),
    Flag("VR_BAND_PACKED", VECREC, "1 (on)", "vecrec",
         "Banded H storage: store only each row's in-band strip (O(R*band_stride)) "
         "instead of the full R*C matrix. Set -DVR_BAND_PACKED=0 to A/B the "
         "memory-traffic saving. No effect on the full-matrix (exact) path."),
    Flag("VR_NO_TRACEBACK", VECREC, "off", "vecrec",
         "Skip traceback, emit trivial CIGAR (timing only; auto-enables VR_UNSAFE)."),
    Flag("VR_DUMMY_PROFILE", VECREC, "off", "vecrec",
         "Constant score profile, skip the O(m*tlen) build (timing only)."),
    Flag("VR_NO_HSTORE", VECREC, "off", "vecrec",
         "Skip the per-chunk H-store memcpy (timing only; implies no traceback)."),
    Flag("VR_NO_ENDPOINT_SCAN", VECREC, "off", "vecrec",
         "Skip the mqe/mte last-row / last-column scans (timing only)."),
    Flag("VR_I8_MAX", VECREC, "120", "vecrec",
         "Upper-bound threshold (min(qlen,tlen)*max_match + end_bonus) for the "
         "per-call I8-vs-I16 dispatch: <= this uses the narrow I8 core. Was "
         "briefly disabled (VR_I8_MAX=0) after a real codegen bug was found -- "
         "both I8 and I16's internal '-infinity' sentinel used only half the "
         "representable range (lowest(elt_type)/2) even though the saturating "
         "arithmetic that motivated that choice makes the full range safe, so a "
         "genuinely bad-but-real score could collide with (and numerically "
         "beat) the sentinel. FIXED upstream in vec-rec (util.py's val_to_str "
         "+ common.h's Sat-variant sentinels now use the type's true "
         "lowest()/max()), which doubles the safe magnitude before collision; "
         "I8 is re-enabled at its original default. VR_I8_SAFE_MAG / "
         "VR_I16_SAFE_MAG (see generate_kernel.py) provide a residual safety "
         "net -- an artificially huge unbanded divergent call can still exceed "
         "even the widened range, and falls back to stock ksw2 in that case. "
         "See end_to_end_baselines/minimap2/CLAUDE.md for the full writeup and "
         "repro."),
]

BY_NAME = {f.name: f for f in FLAGS}


def build_env(defines: list[str]) -> dict[str, str]:
    """Map define tokens (e.g. ['MM_COUNT_MATRIX', 'VR_CHECK=1', 'MM_EXACT_ONLY=0'])
    to the build_minimap2.sh env-var overrides that carry them.

    Unknown flags default to the BOTH (C build) bucket. Returns a dict suitable
    for merging into the build subprocess environment.
    """
    mm: list[str] = []
    vr: list[str] = []
    timing = False
    for d in defines:
        name = d.split("=", 1)[0]
        f = BY_NAME.get(name)
        bucket = f.bucket if f else BOTH
        if bucket == TIMING:
            timing = True
        elif bucket == VECREC:
            vr.append(f"-D{d}")
        else:
            mm.append(f"-D{d}")
    env: dict[str, str] = {}
    if mm:
        env["MM_DEFS"] = " ".join(mm)
    if vr:
        env["VECREC_EXTRA_DEFS"] = " ".join(vr)
    if timing:
        env["VECREC_PHASE_TIMING"] = "1"
    return env


def cache_key(defines: list[str]) -> str:
    """A short, stable suffix identifying a define-set, for naming cached binaries."""
    import hashlib

    if not defines:
        return "default"
    norm = ",".join(sorted(defines))
    return hashlib.sha1(norm.encode()).hexdigest()[:8]
