// SSW (Complete Striped Smith-Waterman) driver -- local alignment only.
// SSW uses SSE2 (via sse2neon on Apple Silicon), so it is SIMD on both x86 and
// ARM.

#include <cstdint>
#include <cstdio>

#include "ssw.h"
#include "ssw_cpp.h"

#include "../common/driver_io.hpp"

using baselines::Input;
using baselines::TimedRun;
using baselines::Timer;

int main(int argc, char** argv) {
    return baselines::run_driver(argc, argv, "ssw_sw", [](const char* path) -> int {
        const Input in = baselines::parse_input(path);

        // Empirically SSW scores a gap of length L as gap_open + L*gap_extend (the
        // open penalty is an additional per-gap surcharge), matching our canonical
        // convention, so the penalties pass through directly. Penalties are positive.
        const uint8_t gap_open = static_cast<uint8_t>(in.gap_open);
        const uint8_t gap_ext = static_cast<uint8_t>(in.gap_extend);
        const uint8_t match = static_cast<uint8_t>(in.match);
        const uint8_t mismatch = static_cast<uint8_t>(-in.mismatch);

        StripedSmithWaterman::Aligner aligner(match, mismatch, gap_open, gap_ext);
        StripedSmithWaterman::Filter filter;
        filter.report_begin_position = true;
        filter.report_cigar = true;

        long score = 0;
        baselines::AlignmentTraceback alignment;
        Timer timer;
        const TimedRun timed = timer.timed_run(in.warmup, in.iterations, [&] {
            StripedSmithWaterman::Alignment aln;
            aligner.Align(in.x.c_str(), in.x.size(), in.y.c_str(), in.y.size(),
                          filter, aln, 15);
            score = aln.sw_score;
            alignment = baselines::sam_cigar_to_traceback(
                aln.cigar, in.x, in.y,
                [](uint32_t packed) -> char { return cigar_int_to_op(packed); });
        });

        baselines::print_result(score, timed, in.cells(), alignment);
        return 0;
    });
}
