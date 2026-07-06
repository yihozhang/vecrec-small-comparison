// Min Cost Climbing Stairs (LeetCode 746): f(i) = cost[i] + min(f(i-1), f(i-2)),
// with base cases f(0)=cost[0], f(1)=cost[1]. In the (min, +) tropical semiring the
// per-step cost rides on both time-varying feedback taps, and a seed stream s carries
// the two base cases; elsewhere s is a large "no-op" value (the TropMin zero, +inf)
// that never wins the min. One long staircase (num_rows defaults to 1).
//
// We use a large FINITE sentinel rather than a literal infinity in the inputs and the
// serial baseline because the build uses -ffast-math (which does not model inf/nan);
// the costs keep every real f(i) far below the sentinel, so min() behaves identically.
#include "bench.hpp"
#include "leetcode_746_min_cost_climbing_stairs.h"

static constexpr double SENTINEL = 1e18;  // stands in for the TropMin zero (+inf) in s

// c[i] = per-step cost (>= 0); s seeds the two base cases and is the +inf sentinel
// elsewhere. Costs are kept non-negative so f(1) = min(c1+c0, c1) = c1. Integer costs
// (stored as double) keep every partial path sum exact (< 2^53) regardless of how the
// dilation reassociates the min-plus products, so correctness is an exact match.
static void build_inputs(double* c, double* s, size_t N, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> cost(0, 100);
  for (size_t i = 0; i < N; ++i) {
    c[i] = (double)cost(rng);
    s[i] = (i < 2) ? c[i] : SENTINEL;
  }
}

static void serial(const double* c, const double* s, double* out, int num_rows, int row_size) {
  const size_t N = (size_t)num_rows * (size_t)row_size;
  for (size_t i = 0; i < N; ++i) {
    double f;
    if (i == 0) f = s[0];                                  // base: f(0) = cost[0]
    else if (i == 1) f = std::min(c[1] + out[0], s[1]);    // base: min(c1+f0, cost1) = cost1
    else f = c[i] + std::min(out[i - 1], out[i - 2]);
    out[i] = f;
  }
}

int main(int argc, char** argv) {
  using K = leetcode_746_min_cost_climbing_stairs;
  using Kb = leetcode_746_min_cost_climbing_stairs_baseline;
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/1, /*cols=*/1 << 22);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<double> c(d.n), s(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n);
  build_inputs(c.data(), s.data(), d.n, cfg.seed);
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(c.data(), s.data(), d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(c.data(), s.data(), d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(c.data(), s.data(), out_serial.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs == 0.0) && (eb.max_abs == 0.0);  // integer-free but exact in double
  bench::report3("leetcode-746-min-cost-climbing-stairs", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
