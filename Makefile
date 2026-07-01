# Build and run the C++ benchmarks. Benchmark source files and binaries are named
# after the kernel (underscored) so they match the generated header generated/<k>.h.
#
#   make <kernel>           # build + run one benchmark (e.g. make second_order_low_pass)
#   make build/<kernel>     # build only
#   make all                # build all
#   make run                # build + run all
#
# Generate the kernel headers first:
#   uv run python scripts/gen_kernels.py --all

# Use clang++ (the generated kernels use clang's ext_vector_type, unsupported by g++).
# `:=` (not `?=`) because make pre-defines CXX=g++ with origin 'default', which `?=`
# would not override. A command-line `make CXX=...` still wins over `:=`.
CXX      := clang++
CXXFLAGS := -O3 -ffast-math -march=native -std=c++20 -Iharness -Igenerated
BUILD    := build

# Kernel names == generated header basenames == benchmark source basenames.
KERNELS := exponential_smoothing double_exp_smoothing box_filter_sat repeated_box_filter \
           first_order_high_pass recursive_gaussian second_order_high_pass \
           second_order_low_pass distance_transform \
           nonuniform_recursive_filter welford_mean continued_fraction

.PHONY: all run clean $(KERNELS)

all: $(addprefix $(BUILD)/,$(KERNELS))

run: $(KERNELS)

# `make <kernel>` builds then runs that benchmark.
$(KERNELS): %: $(BUILD)/%
	./$(BUILD)/$*

$(BUILD):
	mkdir -p $(BUILD)

# Benchmarks with a generated kernel header.
$(addprefix $(BUILD)/,$(KERNELS)): $(BUILD)/%: benchmarks/%.cpp harness/bench.hpp generated/%.h | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

# Helpful error if a kernel header has not been generated yet.
generated/%.h:
	@echo "ERROR: missing $@"; \
	 echo "Generate it with: uv run python scripts/gen_kernels.py $*"; \
	 exit 1

clean:
	rm -rf $(BUILD)
