// SeqAn3 Needleman-Wunsch (global) alignment driver.
// Supports two algo values:
//   "nw"        -- unbanded global alignment (original behaviour)
//   "nw_banded" -- banded global alignment; reads "band_width" from input JSON
//                  and configures seqan3::align_cfg::band_fixed_size with
//                  lower_diagonal = -band_width, upper_diagonal = +band_width.

#include <cstdio>

#include <seqan3/alignment/configuration/align_config_band.hpp>
#include <seqan3/alignment/configuration/align_config_method.hpp>

#include "driver_common.hpp"

int main(int argc, char** argv) {
    return baselines::run_driver(argc, argv, "seqan3_nw", [](const char* path) -> int {
        const baselines::Input in = baselines::parse_input(path);
        if (in.algo == "nw_banded") {
            if (in.band_width < 0) {
                fprintf(stderr, "seqan3_nw: nw_banded requires band_width >= 0 in input JSON\n");
                return 1;
            }
            const auto bw = static_cast<int32_t>(in.band_width);
            auto method = seqan3::align_cfg::method_global{}
                        | seqan3::align_cfg::band_fixed_size{
                              seqan3::align_cfg::lower_diagonal{-bw},
                              seqan3::align_cfg::upper_diagonal{ bw}};
            return seqan3_driver::run(path, method);
        }
        return seqan3_driver::run(path, seqan3::align_cfg::method_global{});
    });
}
