// Min Cost Climbing Stairs (LeetCode 746): f(i) = cost[i] + min(f(i-1), f(i-2)).
// (min, +): the cost rides on both feedback taps; seed s carries f(0)=cost[0], f(1)=cost[1]
// and is a large finite stand-in for the TropMin zero (+inf) elsewhere. int32: path sums stay
// exact and well within INT32_MAX (worst case ~50*N < the INT32_MAX/8 +inf sentinel).
#include "bench.hpp"
#include "leetcode_746_min_cost_climbing_stairs.h"

static constexpr int32_t SENTINEL = INT32_MAX / 8;  // TropMin zero (+inf)

// c[i] = cost (>= 0, integer so min-plus path sums stay exact); s seeds f(0), f(1).
static void build_inputs(int32_t* c, int32_t* s, size_t N, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> cost(0, 100);
  for (size_t i = 0; i < N; ++i) {
    c[i] = (int32_t)cost(rng);
    s[i] = (i < 2) ? c[i] : SENTINEL;
  }
}

static void serial(const int32_t* c, const int32_t* s, int32_t* out, int num_rows, int row_size) {
  const size_t N = (size_t)num_rows * (size_t)row_size;
  for (size_t i = 0; i < N; ++i) {
    int32_t f;
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
  std::vector<int32_t> c(d.n), s(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n);
  build_inputs(c.data(), s.data(), d.n, cfg.seed);
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(c.data(), s.data(), d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(c.data(), s.data(), d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(c.data(), s.data(), out_serial.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs == 0.0) && (eb.max_abs == 0.0);  // int32: exact integer match
  bench::report3("leetcode-746-min-cost-climbing-stairs", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
