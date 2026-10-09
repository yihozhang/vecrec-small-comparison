#!/usr/bin/env bash
# Build two minimap2 binaries from a single clone:
#   bin/minimap2_original  — stock ksw2 extension
#   bin/minimap2_vecrec    — vecrec banded affine extension (USE_VECREC)
#
# Architecture-aware: detects ARM (NEON) vs x86 (SSE) automatically.
# Requires: git, make, clang++, ar (all standard on macOS / Linux).
set -euo pipefail

REPO=$(cd "$(dirname "$0")"/../.. && pwd)
# Single committed minimap2 tree: v2.28 with the VecRec instrumentation, the
# compile-time MM_EXACT_ONLY approx toggle, and the no-banding / no-z-drop /
# no-score-only regime asserts baked into align.c / ksw2_extd2_sse.c. No clone.
MM2="$REPO/end_to_end_baselines/minimap2/minimap2"
GENERATED="$REPO/end_to_end_baselines/minimap2/generated"

if [ ! -d "$MM2" ]; then
    echo "ERROR: vendored minimap2 not found at $MM2" >&2
    exit 1
fi

# ── 2. minimap2 build config (x86_64 only; ARM/NEON is not a target) ──────────
# Only the AFFINE extension (extz2/extd2) is replaced by VecRec. ksw2's original
# SPLICE kernel (exts2) is kept so splice/RNA presets are unchanged.
# On x86 ksw2_dispatch.o defines ALL THREE public symbols (extz2/extd2/exts2),
# so it cannot coexist with the VecRec extz2/extd2 (duplicate symbols). We
# therefore remove every original ext object + the dispatcher, and rebuild a
# standalone non-dispatch exts2 object that exports the public ksw_exts2_sse.
echo "Architecture: x86_64"
MAKE_FLAGS=""
EXT_OBJS="ksw2_extz2_sse41.o ksw2_extd2_sse41.o ksw2_exts2_sse41.o \
          ksw2_extz2_sse2.o ksw2_extd2_sse2.o ksw2_exts2_sse2.o \
          ksw2_dispatch.o"
NEED_EXTS2_REBUILD=1
# Linux: libstdc++ provides the C++ runtime (new/delete) for the VecRec wrapper.
CXXRT="-lstdc++"

# ── 2b. Optional local zlib (for hosts without zlib dev headers) ──────────────
# minimap2's own .c files need zlib.h and -lz. If the system lacks the dev
# package, set ZLIB_DIR to a directory containing zlib.h/zconf.h and a libz.so
# (a symlink to libz.so.1 is fine). Passed through to make as INCLUDES/LIBS so
# we do not rely on $(CPPFLAGS)/$(LDFLAGS) (the Makefile overrides the former
# and the link rule ignores the latter).
ZLIB_INC=""
ZLIB_LD=""
ZLIB_MK_INCLUDES=()
if [ -n "${ZLIB_DIR:-}" ]; then
    ZLIB_INC="-I${ZLIB_DIR}"
    ZLIB_LD="-L${ZLIB_DIR}"
    ZLIB_MK_INCLUDES=("INCLUDES=${ZLIB_INC}")
fi

# ── 3. Generate vecrec kernel (arch-aware) ────────────────────────────────────
# Vector width for schedule selection. Default 256 (epi=32) rather than the
# native 512 (epi=64): minimap2's extension calls are short near-diagonal
# segments (<~128 cols), where the smaller EPI cuts phys_w padding waste and
# fits the DP chunk better — ~18% faster extend on map-ont vs epi=64. Override
# with VECREC_MAX_BITS to experiment.
export VECREC_MAX_BITS="${VECREC_MAX_BITS:-256}"
# Small-matrix-optimized schedules for minimap2's short seed extensions, measured
# at VECREC_MAX_BITS=256 over the candidate schedules: 2-piece idx=12, 1-piece
# idx=9, both ele-per-iter mult=2. generate_kernel.py defaults to element width
# I16; runtime I8/I16 dual-width selection by max score is a later phase.
if [ -n "${1:-}" ]; then
    # Legacy: a single positional arg sets BOTH schedule indices (mult stays 2).
    echo "Generating vecrec kernel header (schedule k${1}, max_bits=${VECREC_MAX_BITS})..."
    cd "$REPO"
    uv run python end_to_end_baselines/scripts/generate_kernel.py "$1"
else
    export VECREC_SCHED_2P="${VECREC_SCHED_2P:-12}"
    export VECREC_SCHED_1P="${VECREC_SCHED_1P:-9}"
    export VECREC_MULT_2P="${VECREC_MULT_2P:-2}"
    export VECREC_MULT_1P="${VECREC_MULT_1P:-2}"
    echo "Generating vecrec kernel header (2p idx=${VECREC_SCHED_2P} 1p idx=${VECREC_SCHED_1P} mult=2, max_bits=${VECREC_MAX_BITS})..."
    cd "$REPO"
    uv run python end_to_end_baselines/scripts/generate_kernel.py
fi

# ── 3b. Debug-build toggle ────────────────────────────────────────────────────
# DEBUG=1 builds -O0 -g binaries with a _debug suffix
# (bin/minimap2_original_debug, bin/minimap2_vecrec_debug) for step-through
# debugging and apples-to-apples init/DP/traceback phase-timing comparison.
# Default (DEBUG unset/0) preserves the optimised build byte-for-byte.
DEBUG="${DEBUG:-0}"
if [ "$DEBUG" = "1" ]; then
    BIN_SUFFIX="_debug"
    CFLAGS_MK=("CFLAGS=-g -O0 -Wall -Wc++-compat")
    VECREC_OPT="-O0 -g"
    EXTS2_OPT="-O0 -g"
    echo "DEBUG build: -O0 -g (binaries get the '${BIN_SUFFIX}' suffix)"
else
    BIN_SUFFIX=""
    CFLAGS_MK=()
    VECREC_OPT="-O3 -march=native"
    EXTS2_OPT="-O3"
fi
# ── 3c. Phase-timing toggle (OFF by default) ──────────────────────────────────
# VECREC_PHASE_TIMING=1 compiles in the [extend-phases] / [vecrec-phases] timers
# and counters. OFF by default: those shared per-call/per-anti-diagonal counters
# contend on cache lines under -t N and (measured) throttle multi-thread scaling
# up to ~5x, so enable ONLY for single-thread profiling — never for a perf run.
# Regime assertions are always compiled in. MM_DEFS carries extra -D flags for
# the C build shared by BOTH binaries (align.c/ksw2), e.g. -DMM_EXACT_ONLY=0
# (restore stock approx-max) or -DMM_COUNT_MATRIX (print [extend-size] at exit).
# See end_to_end_baselines/minimap2/flags.py for the flag registry.
PHASE_DEF=""
CPP_DEFS="-DHAVE_KALLOC ${MM_DEFS:-}"
if [ "${VECREC_PHASE_TIMING:-0}" = "1" ]; then
    PHASE_DEF="-DVECREC_PHASE_TIMING"
    CPP_DEFS="$CPP_DEFS -DVECREC_PHASE_TIMING"
    echo "PHASE-TIMING build: instrumentation ON (single-thread profiling only)"
fi

# ── 3d. Timing-only ablation guard ────────────────────────────────────────────
# The VR_NO_TRACEBACK / VR_DUMMY_PROFILE / VR_NO_HSTORE / VR_NO_ENDPOINT_SCAN
# ablations (in VECREC_EXTRA_DEFS) make the kernel emit garbage alignments, which
# trip minimap2's own CIGAR-span asserts and abort the run before [extend-timing]
# prints. Auto-define VR_UNSAFE (into the shared align.c) so such runs COMPLETE.
# Set VECREC_UNSAFE=1 to force it on. Correct builds never set it.
case " ${VECREC_EXTRA_DEFS:-} " in
    *VR_NO_TRACEBACK*|*VR_DUMMY_PROFILE*|*VR_NO_HSTORE*|*VR_NO_ENDPOINT_SCAN*)
        VECREC_UNSAFE=1 ;;
esac
if [ "${VECREC_UNSAFE:-0}" = "1" ]; then
    CPP_DEFS="$CPP_DEFS -DVR_UNSAFE"
    echo "UNSAFE build: minimap2 CIGAR-span asserts disabled (timing-only ablation)"
fi
CPPFLAGS_MK=("CPPFLAGS=$CPP_DEFS")

# ── 4a. Build minimap2-original ───────────────────────────────────────────────
# Always clean first so libminimap2.a is rebuilt from stock ksw2 sources,
# not from a previously patched archive left by an earlier vecrec build.
echo "Building minimap2-original (clean build)..."
cd "$MM2"
make $MAKE_FLAGS clean 2>/dev/null || true
make $MAKE_FLAGS -j4 "${CFLAGS_MK[@]}" "${CPPFLAGS_MK[@]}" "${ZLIB_MK_INCLUDES[@]}" "LIBS=${ZLIB_LD} -lm -lz -lpthread" 2>/dev/null
cp minimap2 "$REPO/bin/minimap2_original${BIN_SUFFIX}"
echo "  >> bin/minimap2_original${BIN_SUFFIX}  ✓"

# ── 4b. Compile the vecrec ksw2 replacement object ───────────────────────────
echo "Compiling vecrec ksw2 replacement..."
VECREC_OBJ="$MM2/ksw2_vecrec_ext.o"
# -DHAVE_KALLOC: use the real kalloc pool so kmalloc/kfree in the wrapper
# match the pool that minimap2 uses — prevents kfree-of-system-malloc corruption.
clang++ -std=c++20 ${VECREC_OPT} -DUSE_VECREC -DHAVE_KALLOC ${PHASE_DEF} ${VECREC_EXTRA_DEFS:-} \
    -I "$REPO/vec-rec/src/vecrec/templates" \
    -I "$MM2" \
    -c "$GENERATED/ksw2_vecrec_ext.cpp" \
    -o "$VECREC_OBJ"

# ── 4b'. (x86 only) Rebuild a standalone non-dispatch splice object ──────────
# The original ksw_exts2_sse only existed via ksw2_dispatch.o, which we must
# remove (it duplicates extz2/extd2). Recompile ksw2_exts2_sse.c WITHOUT
# KSW_CPU_DISPATCH so it exports the public ksw_exts2_sse directly (native SSE2).
EXTS2_OBJ=""
if [ "${NEED_EXTS2_REBUILD}" = "1" ]; then
    echo "Compiling standalone ksw2 splice object (original splice kernel)..."
    EXTS2_OBJ="$MM2/ksw2_exts2_vecrecbuild.o"
    "${CC:-cc}" ${EXTS2_OPT} -msse2 -DKSW_SSE2_ONLY -DHAVE_KALLOC \
        -I "$MM2" ${ZLIB_INC} \
        -c "$MM2/ksw2_exts2_sse.c" \
        -o "$EXTS2_OBJ"
fi

# ── 4b''. Renamed stock ksw2 affine objects (for approx-max delegation) ───────
# The vecrec wrapper delegates KSW_EZ_APPROX_MAX calls to the ORIGINAL ksw2
# affine kernels (vecrec has no approximate-max mode). We compile ksw2's affine
# sources non-dispatch (KSW_SSE2_ONLY => defines the public symbol directly) but
# rename that public symbol to *_orig via -D, so both the vecrec public symbol
# and the stock fallback coexist in the archive without collision.
echo "Compiling renamed ksw2 affine objects (approx-max fallback)..."
EXTD2_ORIG_OBJ="$MM2/ksw2_extd2_orig.o"
EXTZ2_ORIG_OBJ="$MM2/ksw2_extz2_orig.o"
ORIG_CC_FLAGS="-msse2 -DKSW_SSE2_ONLY"
"${CC:-cc}" ${EXTS2_OPT} ${ORIG_CC_FLAGS} -DHAVE_KALLOC ${PHASE_DEF} -Dksw_extd2_sse=ksw_extd2_sse_orig \
    -I "$MM2" ${ZLIB_INC} -c "$MM2/ksw2_extd2_sse.c" -o "$EXTD2_ORIG_OBJ"
"${CC:-cc}" ${EXTS2_OPT} ${ORIG_CC_FLAGS} -DHAVE_KALLOC ${PHASE_DEF} -Dksw_extz2_sse=ksw_extz2_sse_orig \
    -I "$MM2" ${ZLIB_INC} -c "$MM2/ksw2_extz2_sse.c" -o "$EXTZ2_ORIG_OBJ"

# ── 4c. Build minimap2-vecrec: patch libminimap2.a then re-link ──────────────
echo "Building minimap2-vecrec..."
cd "$MM2"

# Remove the affine (and, on x86, dispatch) ksw2 objects from the archive and
# add our replacements. The original splice kernel is preserved: on ARM by
# keeping ksw2_exts2_neon.o, on x86 by adding the standalone exts2 object above.
# ar d is idempotent: silently ignores members not in the archive.
for OBJ in $EXT_OBJS; do
    ar d libminimap2.a "$OBJ" 2>/dev/null || true
done
ar r libminimap2.a "$VECREC_OBJ"
[ -n "$EXTS2_OBJ" ] && ar r libminimap2.a "$EXTS2_OBJ"
# Renamed stock affine kernels for approx-max delegation (ksw_ext{d,z}2_sse_orig).
ar r libminimap2.a "$EXTD2_ORIG_OBJ" "$EXTZ2_ORIG_OBJ"

# Force re-link by removing the binary; add the C++ runtime (new/delete) — its
# name is arch-dependent (libc++ on macOS, libstdc++ on Linux), see CXXRT above.
rm -f minimap2
make $MAKE_FLAGS -j4 minimap2 "${CFLAGS_MK[@]}" "${CPPFLAGS_MK[@]}" "${ZLIB_MK_INCLUDES[@]}" "LIBS=${ZLIB_LD} -lm -lz -lpthread $CXXRT" 2>/dev/null

cp minimap2 "$REPO/bin/minimap2_vecrec${BIN_SUFFIX}"
echo "  >> bin/minimap2_vecrec${BIN_SUFFIX}  ✓"

echo ""
echo "Done.  Run:  python end_to_end_baselines/scripts/run_e2e.py"
