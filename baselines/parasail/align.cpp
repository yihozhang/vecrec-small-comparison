// parasail driver: dispatches to striped, diag (antidiagonal), scan (prefix-sum),
// or scalar stats implementations via the ``algo`` field in input.json.

#include <cstdio>

#include <parasail.h>

#include "../common/driver_io.hpp"
#include "driver_common.hpp"

using baselines::Input;

int main(int argc, char** argv) {
    return baselines::run_driver(argc, argv, "parasail_align", [](const char* path) -> int {
        const Input in = baselines::parse_input(path);
        return parasail_driver::run_align(in);
    });
}
