# ksw2: SIMD-only kernel comparison. Exposes the vectorized ksw2 kernels
# (ksw_gg2_sse, ksw_extz2_sse, ksw_extd2_sse, ksw_extf2_sse, ksw_ll_i16); no
# scalar variants are built. SIMD intrinsics are translated to NEON via sse2neon
# on Apple Silicon.  Build flags mirror what minimap2's Makefile uses for the ARM
# NEON target:
#   -DKSW_SSE2_ONLY  — disables SSE4.1 code paths
#   -D__SSE2__       — enables the SSE2 branch in the source
#   -DHAVE_KALLOC    — required by ksw2 for its internal allocator
set(KSW2_SRC "${THIRD_PARTY}/ksw2")
# minimap2 lives as a single committed tree under end_to_end_baselines/ (no
# longer a gitignored third_party clone). On x86 this baseline only pulls the
# unpatched ksw2_ll_sse.c from it.
set(MM2_SRC  "${THIRD_PARTY}/../end_to_end_baselines/minimap2/minimap2")

# The SIMD source (ksw2_extd2_sse.c) and kalloc live in the ksw2 repo itself.
# On ARM, the SSE2 intrinsics it uses need minimap2's bundled sse2neon shim to
# translate to NEON, so we pull those two .c files (and the include dir) from
# the minimap2 clone. On x86_64 we have native SSE2, so ksw2's own sources
# compile directly — no minimap2 checkout required (KSW_SSE2_ONLY + __SSE2__,
# without USE_SIMDE, selects native <emmintrin.h>).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(KSW2_SIMD_DIR "${MM2_SRC}")
    set(KSW2_SIMD_INCLUDES "${KSW2_SRC}" "${MM2_SRC}" "${MM2_SRC}/sse2neon")
    # ARM: sse2neon translates the SSE2 branch to NEON; KSW_SSE2_ONLY disables the
    # x86 SSE4.1 code paths that have no NEON translation here.
    set(KSW2_DEFS HAVE_KALLOC KSW_SSE2_ONLY __SSE2__)
    set(KSW2_OPTS "")
else()
    set(KSW2_SIMD_DIR "${KSW2_SRC}")
    set(KSW2_SIMD_INCLUDES "${KSW2_SRC}")
    # x86: build a FAIR baseline. -march=native enables the SSE4.1 fast path
    # (native signed int8 _mm_max_epi8 in extd2, etc.) plus AVX2/AVX-512 auto-
    # vectorization, matching the VecRec driver's -march=native. Do NOT set
    # KSW_SSE2_ONLY (that #undef's __SSE4_1__ and forces the slow SSE2 emulation).
    set(KSW2_DEFS HAVE_KALLOC)
    set(KSW2_OPTS -march=native)
endif()

# Source locations differ across the two vendored trees:
#   - ksw2_extz2_sse.c / ksw2_extd2_sse.c: present in both (ARM path prefers the
#     sse2neon-friendly minimap2 copy, so they come from KSW2_SIMD_DIR).
#   - ksw2_gg2_sse.c / ksw2_extf2_sse.c:   only in the ksw2 tree.
#   - ksw2_ll_sse.c:                       only in the minimap2 tree.
add_executable(ksw2_align
    align.cpp
    "${KSW2_SIMD_DIR}/ksw2_extz2_sse.c"
    "${KSW2_SIMD_DIR}/ksw2_extd2_sse.c"
    "${KSW2_SRC}/ksw2_gg2_sse.c"
    "${KSW2_SRC}/ksw2_extf2_sse.c"
    "${MM2_SRC}/ksw2_ll_sse.c"
    "${KSW2_SIMD_DIR}/kalloc.c"
)
target_include_directories(ksw2_align PRIVATE ${KSW2_SIMD_INCLUDES})
target_compile_definitions(ksw2_align PRIVATE ${KSW2_DEFS})
if(KSW2_OPTS)
    target_compile_options(ksw2_align PRIVATE ${KSW2_OPTS})
endif()
target_link_libraries(ksw2_align PRIVATE baselines_common)
