// Hand-written dilated NW v2 driver — register-only prefix-max approach.
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "driver_io.hpp"
#include "matrix_traceback.hpp"
#include "nw_dilated_v2.hpp"

using baselines::Input;
using baselines::Timer;

int main(int argc, char** argv) {
    return baselines::run_driver(argc, argv, "nw_dilated_v2", [](const char* path) -> int {
        const Input in = baselines::parse_input(path);
        const int rows = (int)in.x.size();
        const int cols = (int)in.y.size();
        const long cells = (long)rows * (long)cols;
        const int gap   = -static_cast<int>(in.gap_extend);
        const int match = static_cast<int>(in.match);
        const int mm    = static_cast<int>(in.mismatch);

        const int phys_w = ((cols + 1 + EPI_V2 - 1) / EPI_V2) * EPI_V2;
        const int N = (rows + 1) * phys_w;
        std::vector<int16_t> output(static_cast<size_t>(N), 0);

        long score = 0;
        baselines::AlignmentTraceback alignment;
        Timer timer;

        auto timed = timer.timed_run(in.warmup, in.iterations, [&] {
            std::fill(output.begin(), output.end(), int16_t(0));
            score = nw_dilated_v2(output.data(), phys_w, in.x, in.y, match, mm, gap);
        });

        std::vector<char> q(static_cast<size_t>(rows + cols + 2));
        std::vector<char> r(static_cast<size_t>(rows + cols + 2));
        std::vector<char> c(static_cast<size_t>(rows + cols + 2));
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
