# WFA2-lib (wavefront alignment): compile the library sources directly so we
# avoid WFA2's top-level CMake (PkgConfig + align_benchmark + tests).
set(WFA2_DIR "${THIRD_PARTY}/wfa2")

set(WFA2_SOURCES
  "${WFA2_DIR}/wavefront/wavefront_align.c"
  "${WFA2_DIR}/wavefront/wavefront_aligner.c"
  "${WFA2_DIR}/wavefront/wavefront_attributes.c"
  "${WFA2_DIR}/wavefront/wavefront_backtrace_buffer.c"
  "${WFA2_DIR}/wavefront/wavefront_backtrace.c"
  "${WFA2_DIR}/wavefront/wavefront_backtrace_offload.c"
  "${WFA2_DIR}/wavefront/wavefront_bialign.c"
  "${WFA2_DIR}/wavefront/wavefront_bialigner.c"
  "${WFA2_DIR}/wavefront/wavefront.c"
  "${WFA2_DIR}/wavefront/wavefront_components.c"
  "${WFA2_DIR}/wavefront/wavefront_compute_affine2p.c"
  "${WFA2_DIR}/wavefront/wavefront_compute_affine.c"
  "${WFA2_DIR}/wavefront/wavefront_compute.c"
  "${WFA2_DIR}/wavefront/wavefront_compute_edit.c"
  "${WFA2_DIR}/wavefront/wavefront_compute_linear.c"
  "${WFA2_DIR}/wavefront/wavefront_debug.c"
  "${WFA2_DIR}/wavefront/wavefront_display.c"
  "${WFA2_DIR}/wavefront/wavefront_extend.c"
  "${WFA2_DIR}/wavefront/wavefront_heuristic.c"
  "${WFA2_DIR}/wavefront/wavefront_pcigar.c"
  "${WFA2_DIR}/wavefront/wavefront_penalties.c"
  "${WFA2_DIR}/wavefront/wavefront_plot.c"
  "${WFA2_DIR}/wavefront/wavefront_sequences.c"
  "${WFA2_DIR}/wavefront/wavefront_slab.c"
  "${WFA2_DIR}/wavefront/wavefront_unialign.c"
  "${WFA2_DIR}/wavefront/wavefront_termination.c"
  "${WFA2_DIR}/wavefront/wavefront_extend_kernels_avx.c"
  "${WFA2_DIR}/wavefront/wavefront_extend_kernels.c"
  "${WFA2_DIR}/system/mm_stack.c"
  "${WFA2_DIR}/system/mm_allocator.c"
  "${WFA2_DIR}/system/profiler_counter.c"
  "${WFA2_DIR}/system/profiler_timer.c"
  "${WFA2_DIR}/utils/bitmap.c"
  "${WFA2_DIR}/utils/dna_text.c"
  "${WFA2_DIR}/utils/sequence_buffer.c"
  "${WFA2_DIR}/utils/vector.c"
  "${WFA2_DIR}/utils/commons.c"
  "${WFA2_DIR}/utils/heatmap.c"
  "${WFA2_DIR}/alignment/affine2p_penalties.c"
  "${WFA2_DIR}/alignment/affine_penalties.c"
  "${WFA2_DIR}/alignment/cigar.c"
  "${WFA2_DIR}/alignment/score_matrix.c"
)

add_library(wfa2_static STATIC ${WFA2_SOURCES})
set_target_properties(wfa2_static PROPERTIES OUTPUT_NAME wfa2_baseline)
target_include_directories(wfa2_static PUBLIC "${WFA2_DIR}")
# WFA2 is C; keep it quiet under our C++ driver flags.
target_compile_options(wfa2_static PRIVATE -w)
target_link_libraries(wfa2_static PUBLIC m)

add_executable(wfa2_align align.cpp)
target_include_directories(wfa2_align PRIVATE "${WFA2_DIR}")
target_link_libraries(wfa2_align PRIVATE baselines_common wfa2_static)
