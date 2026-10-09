// Hand-written dilated NW driver.
// Uses the striped (dilated) recurrence from nw_dilated.hpp:
//   Stage 1: embarrassingly parallel main pass (diag + above, all lanes independent)
//   Stage 2: LANES-1 = 7 correction passes for left-gap propagation
// No struct overhead, no InlineAsm fences, no intermediate memory spill.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "driver_io.hpp"
#include "matrix_traceback.hpp"
#include "nw_dilated.hpp"

using baselines::Input;
using baselines::Timer;

int main(int argc, char** argv) {
    return baselines::run_driver(argc, argv, "nw_dilated", [](const char* path) -> int {
        const Input in = baselines::parse_input(path);
        const int rows = static_cast<int>(in.x.size());
        const int cols = static_cast<int>(in.y.size());
        const long cells = static_cast<long>(rows) * static_cast<long>(cols);
        const int gap    = -static_cast<int>(in.gap_extend);
        const int match  = static_cast<int>(in.match);
        const int mm     = static_cast<int>(in.mismatch);

        // Physical row width rounded up to LANES so the SIMD loop never reads
        // past the end of the buffer.
        const int phys_w = ((cols + 1 + LANES - 1) / LANES) * LANES;
        const int N      = (rows + 1) * phys_w;

        std::vector<int16_t> output(static_cast<std::size_t>(N), 0);

        long score = 0;
        baselines::AlignmentTraceback alignment;
        Timer timer;

        auto timed = timer.timed_run(in.warmup, in.iterations, [&] {
            std::fill(output.begin(), output.end(), int16_t(0));
            score = nw_dilated(output.data(), phys_w,
                               in.x, in.y, match, mm, gap);
        });

        // Traceback over the full matrix.
        std::vector<char> q(static_cast<std::size_t>(rows + cols + 2));
        std::vector<char> r(static_cast<std::size_t>(rows + cols + 2));
        std::vector<char> c(static_cast<std::size_t>(rows + cols + 2));
        baselines::nw_traceback_full_inline(
            output.data(), rows, cols, phys_w,
            in.x.c_str(), in.y.c_str(), gap, match, mm,
            q.data(), r.data(), c.data());
        alignment.query = q.data();
        alignment.ref   = r.data();
        alignment.comp  = c.data();

        baselines::print_result(score, timed, cells, alignment);
        return 0;
    });
}
