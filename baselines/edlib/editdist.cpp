// edlib driver: global (Needleman-Wunsch mode) unit-cost edit distance.
//
// edlib computes Levenshtein edit distance (substitution/insertion/deletion all
// cost 1); it does not use a scoring matrix or gap penalties, so the scoring
// fields of the input are ignored. The reported "score" is the edit distance.

#include <cstdio>

#include "edlib.h"

#include "../common/driver_io.hpp"

using baselines::Input;
using baselines::TimedRun;
using baselines::Timer;

int main(int argc, char** argv) {
    return baselines::run_driver(argc, argv, "edlib_editdist", [](const char* path) -> int {
        const Input in = baselines::parse_input(path);

        const EdlibAlignConfig cfg = edlibNewAlignConfig(
            -1, EDLIB_MODE_NW, EDLIB_TASK_PATH, nullptr, 0);

        long score = 0;
        baselines::AlignmentTraceback alignment;
        Timer timer;
        const TimedRun timed = timer.timed_run(in.warmup, in.iterations, [&] {
            EdlibAlignResult r = edlibAlign(
                in.x.c_str(), static_cast<int>(in.x.size()),
                in.y.c_str(), static_cast<int>(in.y.size()), cfg);
            score = r.editDistance;
            alignment = baselines::edlib_path_to_traceback(
                r.alignment, r.alignmentLength, in.x, in.y);
            edlibFreeAlignResult(r);
        });

        baselines::print_result(score, timed, in.cells(), alignment);
        return 0;
    });
}
