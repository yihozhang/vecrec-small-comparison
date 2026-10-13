#!/usr/bin/env bash
# Build two minimap2 binaries from a single clone:
#   bin/minimap2_original  — stock ksw2 extension
#   bin/minimap2_vecrec    — vecrec banded affine extension (USE_VECREC)
#
# Architecture-aware: x86 (SSE) or arm64 (NEON via sse2neon), picked by uname -m.
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

# ── 2. minimap2 build config (x86_64 SSE or arm64 NEON via sse2neon) ─────────
# Only the AFFINE extension (extz2/extd2) is replaced by VecRec. ksw2's original
# SPLICE kernel (exts2) is kept so splice/RNA presets are unchanged.
# x86: ksw2_dispatch.o defines ALL THREE public symbols (extz2/extd2/exts2),
#   so it cannot coexist with the VecRec extz2/extd2 (duplicate symbols). We
#   therefore remove every original ext object + the dispatcher, and rebuild a
#   standalone non-dispatch exts2 object that exports the public ksw_exts2_sse.
# arm64: minimap2's Makefile (aarch64=1) builds the three *_neon.o objects
#   non-dispatch through sse2neon, so we drop only the two affine ones and keep
#   ksw2_exts2_neon.o as is. ORIG_CC_FLAGS mirrors the Makefile's NEON recipe
#   for the renamed stock affine objects compiled in step 4b''.
ARCH="$(uname -m)"
case "$ARCH" in
    arm64|aarch64)
        echo "Architecture: ${ARCH} (NEON via sse2neon)"
        MAKE_FLAGS="aarch64=1"
        EXT_OBJS="ksw2_extz2_neon.o ksw2_extd2_neon.o"
        NEED_EXTS2_REBUILD=0
        ORIG_CC_FLAGS="-DKSW_SSE2_ONLY -D__SSE2__ -I$MM2/sse2neon -D_FILE_OFFSET_BITS=64 -fsigned-char"
        EXTS2_ARCH_FLAGS=""
        # The Makefile appends these via `CFLAGS+=`, which a command-line CFLAGS
        # overrides, so they are re-added to MM2_CFLAGS below.
        MM2_ARCH_CFLAGS="-D_FILE_OFFSET_BITS=64 -fsigned-char"
        ;;
    *)
        echo "Architecture: ${ARCH} (SSE)"
        MAKE_FLAGS=""
        EXT_OBJS="ksw2_extz2_sse41.o ksw2_extd2_sse41.o ksw2_exts2_sse41.o \
                  ksw2_extz2_sse2.o ksw2_extd2_sse2.o ksw2_exts2_sse2.o \
                  ksw2_dispatch.o"
        NEED_EXTS2_REBUILD=1
        # Standalone (non-dispatch) stock ksw2 objects. Without KSW_SSE2_ONLY they
        # take the SSE4.1 code path whenever the target supports it, which is
        # the path the stock binary's runtime dispatcher selects on such CPUs.
        ORIG_CC_FLAGS=""
        EXTS2_ARCH_FLAGS=""
        MM2_ARCH_CFLAGS=""
        ;;
esac
# Link the C++ runtime used by clang++ for the VecRec wrapper.
case "$(uname -s)" in
    Darwin) CXXRT="-lc++" ;;
    *)      CXXRT="-lstdc++" ;;
esac

# ── 2b. Optional local zlib (for hosts without zlib dev headers) ──────────────
# minimap2's own .c files need zlib.h and -lz. If the system lacks the dev
# package, set ZLIB_DIR to a directory containing zlib.h/zconf.h and a libz.so
# (a symlink to libz.so.1 is fine). Passed through to make as INCLUDES/LIBS so
# we do not rely on $(CPPFLAGS)/$(LDFLAGS) (the Makefile overrides the former
# and the link rule ignores the latter).
ZLIB_INC=""
ZLIB_LD=""
if [ -n "${ZLIB_DIR:-}" ]; then
    ZLIB_INC="-I${ZLIB_DIR}"
    ZLIB_LD="-L${ZLIB_DIR}"
fi

# ── 2c. Preflight: minimap2 needs zlib.h and -lz ─────────────────────────────
if ! printf '#include <zlib.h>\nint main(void){return 0;}\n' | "${CC:-cc}" -x c - -o /dev/null ${ZLIB_INC} ${ZLIB_LD} -lz 2>/dev/null; then
    echo "ERROR: zlib development files not found (zlib.h / libz)." >&2
    echo "       Debian/Ubuntu: sudo apt install zlib1g-dev   (macOS: included in the SDK)" >&2
    echo "       or point ZLIB_DIR at a directory containing zlib.h, zconf.h and libz.so/.a" >&2
    exit 1
fi

# ── 3. Generate vecrec kernel (arch-aware) ────────────────────────────────────
# Vector width for schedule selection. Default 256 (epi=32) rather than the
# native 512 (epi=64). Override with VECREC_MAX_BITS.
export VECREC_MAX_BITS="${VECREC_MAX_BITS:-256}"
# Search VecRec's candidate schedules and codegen settings on this host for
# each 1-piece/2-piece and I8/I16 kernel. A positional index is retained for
# explicit experiments; VECREC_SCHED_* can override individual variants.
if [ -n "${1:-}" ]; then
    echo "Generating vecrec kernel header (schedule k${1}, max_bits=${VECREC_MAX_BITS})..."
    cd "$REPO"
    uv run python end_to_end_baselines/scripts/generate_kernel.py "$1"
else
    echo "Searching VecRec schedules (max_bits=${VECREC_MAX_BITS}) and generating the drop-in..."
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
    VECREC_OPT="-O0 -g"
    EXTS2_OPT="-O0 -g"
    MM2_CFLAGS="-g -O0 -Wall -Wc++-compat ${MM2_ARCH_CFLAGS}"
    echo "DEBUG build: -O0 -g (binaries get the '${BIN_SUFFIX}' suffix)"
else
    BIN_SUFFIX=""
    VECREC_OPT="-O3 -march=native"
    # Stock minimap2/ksw2 gets the same optimisation level and target as the
    # VecRec wrapper (minimap2's Makefile default is -O2 with no -march).
    EXTS2_OPT="-O3 -march=native"
    MM2_CFLAGS="-g -Wall -O3 -march=native -Wc++-compat ${MM2_ARCH_CFLAGS}"
fi
# ── 3c. Phase-timing toggle (OFF by default) ──────────────────────────────────
# VECREC_PHASE_TIMING=1 compiles in the [extend-phases] / [vecrec-phases] timers
# and counters. OFF by default: those shared per-call/per-anti-diagonal counters
# contend on cache lines under -t N, so enable ONLY for single-thread profiling —
# never for a perf run.
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
# Keep this array nonempty: macOS's Bash 3.2 treats expansion of an empty array
# as an unbound variable under `set -u`.
MAKE_ARGS=(-j4 "CPPFLAGS=$CPP_DEFS" "CFLAGS=$MM2_CFLAGS")
if [ -n "$ZLIB_INC" ]; then
    MAKE_ARGS+=("INCLUDES=$ZLIB_INC")
fi

# ── 4a. Build minimap2-original ───────────────────────────────────────────────
# Always clean first so libminimap2.a is rebuilt from stock ksw2 sources,
# not from a previously patched archive left by an earlier vecrec build.
echo "Building minimap2-original (clean build)..."
cd "$MM2"
make $MAKE_FLAGS clean 2>/dev/null || true
make $MAKE_FLAGS "${MAKE_ARGS[@]}" "LIBS=${ZLIB_LD} -lm -lz -lpthread"
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
# KSW_CPU_DISPATCH so it exports the public ksw_exts2_sse directly.
EXTS2_OBJ=""
if [ "${NEED_EXTS2_REBUILD}" = "1" ]; then
    echo "Compiling standalone ksw2 splice object (original splice kernel)..."
    EXTS2_OBJ="$MM2/ksw2_exts2_vecrecbuild.o"
    "${CC:-cc}" ${EXTS2_OPT} ${EXTS2_ARCH_FLAGS} -DHAVE_KALLOC \
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
make $MAKE_FLAGS "${MAKE_ARGS[@]}" minimap2 "LIBS=${ZLIB_LD} -lm -lz -lpthread $CXXRT"

cp minimap2 "$REPO/bin/minimap2_vecrec${BIN_SUFFIX}"
echo "  >> bin/minimap2_vecrec${BIN_SUFFIX}  ✓"

echo ""
echo "Done.  Run:  python end_to_end_baselines/scripts/run_e2e.py"
