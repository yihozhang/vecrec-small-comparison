// Shared parasail dispatch: maps harness algo tags to parasail entry points.
//
// Algo tags look like ``sw_striped_16``, ``nw_diag_16``, ``nw_scan_16``,
// ``sw_stats``, etc.  Each tag selects one of parasail's vectorization
// families (striped, diag/antidiagonal, scan/prefix-sum, or scalar stats).

#pragma once

#include <parasail.h>

#include <stdexcept>
#include <string>
#include <utility>

#include "../common/driver_io.hpp"

namespace parasail_driver {

using AlignFn = parasail_result_t* (*)(const char*, int, const char*, int, int, int,
                                       const parasail_matrix_t*);

inline std::string trace_algo_tag(const std::string& algo) {
    if (algo == "nw_stats") return "nw_trace";
    if (algo == "sw_stats") return "sw_trace";
    if (algo == "nw") return "nw_trace_striped_16";
    if (algo == "sw") return "sw_trace_striped_16";
    if (algo.rfind("nw_", 0) == 0) return "nw_trace_" + algo.substr(3);
    if (algo.rfind("sw_", 0) == 0) return "sw_trace_" + algo.substr(3);
    throw std::runtime_error("unknown parasail algo tag: " + algo);
}

inline AlignFn resolve(const std::string& algo) {
    // Scalar reference (always correct; used as a correctness anchor).
    if (algo == "nw_stats") return parasail_nw_stats;
    if (algo == "sw_stats") return parasail_sw_stats;

    // Striped vectorization.
    if (algo == "nw_striped_8") return parasail_nw_striped_8;
    if (algo == "sw_striped_8") return parasail_sw_striped_8;
    if (algo == "nw_striped_16") return parasail_nw_striped_16;
    if (algo == "sw_striped_16") return parasail_sw_striped_16;
    if (algo == "nw_striped_32") return parasail_nw_striped_32;
    if (algo == "sw_striped_32") return parasail_sw_striped_32;
    if (algo == "nw_striped_64") return parasail_nw_striped_64;
    if (algo == "sw_striped_64") return parasail_sw_striped_64;
    if (algo == "nw_striped_sat") return parasail_nw_striped_sat;
    if (algo == "sw_striped_sat") return parasail_sw_striped_sat;

    // Antidiagonal (parasail calls this "diag").
    if (algo == "nw_diag_8") return parasail_nw_diag_8;
    if (algo == "sw_diag_8") return parasail_sw_diag_8;
    if (algo == "nw_diag_16") return parasail_nw_diag_16;
    if (algo == "sw_diag_16") return parasail_sw_diag_16;
    if (algo == "nw_diag_32") return parasail_nw_diag_32;
    if (algo == "sw_diag_32") return parasail_sw_diag_32;
    if (algo == "nw_diag_64") return parasail_nw_diag_64;
    if (algo == "sw_diag_64") return parasail_sw_diag_64;
    if (algo == "nw_diag_sat") return parasail_nw_diag_sat;
    if (algo == "sw_diag_sat") return parasail_sw_diag_sat;

    // Prefix-sum / scan vectorization.
    if (algo == "nw_scan_8") return parasail_nw_scan_8;
    if (algo == "sw_scan_8") return parasail_sw_scan_8;
    if (algo == "nw_scan_16") return parasail_nw_scan_16;
    if (algo == "sw_scan_16") return parasail_sw_scan_16;
    if (algo == "nw_scan_32") return parasail_nw_scan_32;
    if (algo == "sw_scan_32") return parasail_sw_scan_32;
    if (algo == "nw_scan_64") return parasail_nw_scan_64;
    if (algo == "sw_scan_64") return parasail_sw_scan_64;
    if (algo == "nw_scan_sat") return parasail_nw_scan_sat;
    if (algo == "sw_scan_sat") return parasail_sw_scan_sat;

    // Backward-compatible aliases (old cases used bare ``sw`` / ``nw``).
    if (algo == "nw") return parasail_nw_striped_16;
    if (algo == "sw") return parasail_sw_striped_16;

    throw std::runtime_error("unknown parasail algo tag: " + algo);
}

inline AlignFn resolve_trace(const std::string& algo) {
    if (algo == "nw_trace") return parasail_nw_trace;
    if (algo == "sw_trace") return parasail_sw_trace;

    if (algo == "nw_trace_striped_8") return parasail_nw_trace_striped_8;
    if (algo == "sw_trace_striped_8") return parasail_sw_trace_striped_8;
    if (algo == "nw_trace_striped_16") return parasail_nw_trace_striped_16;
    if (algo == "sw_trace_striped_16") return parasail_sw_trace_striped_16;
    if (algo == "nw_trace_striped_32") return parasail_nw_trace_striped_32;
    if (algo == "sw_trace_striped_32") return parasail_sw_trace_striped_32;
    if (algo == "nw_trace_striped_64") return parasail_nw_trace_striped_64;
    if (algo == "sw_trace_striped_64") return parasail_sw_trace_striped_64;
    if (algo == "nw_trace_striped_sat") return parasail_nw_trace_striped_sat;
    if (algo == "sw_trace_striped_sat") return parasail_sw_trace_striped_sat;

    if (algo == "nw_trace_diag_8") return parasail_nw_trace_diag_8;
    if (algo == "sw_trace_diag_8") return parasail_sw_trace_diag_8;
    if (algo == "nw_trace_diag_16") return parasail_nw_trace_diag_16;
    if (algo == "sw_trace_diag_16") return parasail_sw_trace_diag_16;
    if (algo == "nw_trace_diag_32") return parasail_nw_trace_diag_32;
    if (algo == "sw_trace_diag_32") return parasail_sw_trace_diag_32;
    if (algo == "nw_trace_diag_64") return parasail_nw_trace_diag_64;
    if (algo == "sw_trace_diag_64") return parasail_sw_trace_diag_64;
    if (algo == "nw_trace_diag_sat") return parasail_nw_trace_diag_sat;
    if (algo == "sw_trace_diag_sat") return parasail_sw_trace_diag_sat;

    if (algo == "nw_trace_scan_8") return parasail_nw_trace_scan_8;
    if (algo == "sw_trace_scan_8") return parasail_sw_trace_scan_8;
    if (algo == "nw_trace_scan_16") return parasail_nw_trace_scan_16;
    if (algo == "sw_trace_scan_16") return parasail_sw_trace_scan_16;
    if (algo == "nw_trace_scan_32") return parasail_nw_trace_scan_32;
    if (algo == "sw_trace_scan_32") return parasail_sw_trace_scan_32;
    if (algo == "nw_trace_scan_64") return parasail_nw_trace_scan_64;
    if (algo == "sw_trace_scan_64") return parasail_sw_trace_scan_64;
    if (algo == "nw_trace_scan_sat") return parasail_nw_trace_scan_sat;
    if (algo == "sw_trace_scan_sat") return parasail_sw_trace_scan_sat;

    throw std::runtime_error("unknown parasail trace algo tag: " + algo);
}

inline std::pair<long, baselines::AlignmentTraceback> align_with_traceback(
    const baselines::Input& in, parasail_matrix_t* matrix) {
    AlignFn fn = resolve_trace(trace_algo_tag(in.algo));
    parasail_result_t* r =
        fn(in.x.c_str(), static_cast<int>(in.x.size()), in.y.c_str(),
           static_cast<int>(in.y.size()), static_cast<int>(in.gap_open + in.gap_extend),
           static_cast<int>(in.gap_extend), matrix);
    const long score = parasail_result_get_score(r);
    parasail_traceback_t* tb = parasail_result_get_traceback(
        r, in.x.c_str(), static_cast<int>(in.x.size()), in.y.c_str(),
        static_cast<int>(in.y.size()), matrix, '|', '+', '-');
    baselines::AlignmentTraceback out;
    if (tb != nullptr) {
        if (tb->query != nullptr) out.query = tb->query;
        if (tb->ref != nullptr) out.ref = tb->ref;
        if (tb->comp != nullptr) out.comp = tb->comp;
        parasail_traceback_free(tb);
    }
    parasail_result_free(r);
    return {score, out};
}

// Scalar banded NW (no SIMD, no traceback -- parasail has no
// nw_trace_banded). ``k`` is a symmetric diagonal half-width: for
// equal-length sequences (src/nw_banded.c: low = s1Len - s2Len - k = -k when
// s1Len == s2Len) the band spans offsets [-k, +k], exactly matching this
// harness's ``band_width`` convention, so k == band_width directly.
inline int run_align_banded(const baselines::Input& in, parasail_matrix_t* matrix) {
    if (in.band_width < 0) {
        throw std::runtime_error("parasail nw_banded requires band_width >= 0 in input JSON");
    }
    const int open = static_cast<int>(in.gap_open + in.gap_extend);
    const int gap = static_cast<int>(in.gap_extend);
    const int k = static_cast<int>(in.band_width);

    long score = 0;
    baselines::Timer timer;
    const baselines::TimedRun timed = timer.timed_run(in.warmup, in.iterations, [&] {
        parasail_result_t* r = parasail_nw_banded(
            in.x.c_str(), static_cast<int>(in.x.size()),
            in.y.c_str(), static_cast<int>(in.y.size()), open, gap, k, matrix);
        score = parasail_result_get_score(r);
        parasail_result_free(r);
    });
    baselines::print_result(score, timed, in.cells());
    return 0;
}

inline int run_align(const baselines::Input& in) {
    parasail_matrix_t* matrix = parasail_matrix_create(
        "ACGT", static_cast<int>(in.match), static_cast<int>(in.mismatch));

    if (in.algo == "nw_banded") {
        const int rc = run_align_banded(in, matrix);
        parasail_matrix_free(matrix);
        return rc;
    }

    long score = 0;
    baselines::AlignmentTraceback alignment;
    baselines::Timer timer;
    const baselines::TimedRun timed = timer.timed_run(in.warmup, in.iterations, [&] {
        std::tie(score, alignment) = align_with_traceback(in, matrix);
    });

    parasail_matrix_free(matrix);
    baselines::print_result(score, timed, in.cells(), alignment);
    return 0;
}

}  // namespace parasail_driver
