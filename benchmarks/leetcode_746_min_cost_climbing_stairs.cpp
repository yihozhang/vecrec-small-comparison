// Min Cost Climbing Stairs (LeetCode 746): dp(i) = min(dp(i-1)+cost[i-1], dp(i-2)+cost[i-2]).
// (min, +): you stand for free on step 0 or 1 (dp(0)=dp(1)=0) and pay cost[k] to *leave* step k,
// so the cost rides on the source/lag taps (cost[i-1] on f(i-1), cost[i-2] on f(i-2)). The base
// dp(0)=dp(1)=0 is injected structurally as an impulse of the TropMin one (0) at positions 0 and 1
// -- no seed input array. int32: min-plus path sums stay exact and bounded (worst case ~50*N,
// well below the INT32_MAX/8 +inf sentinel the kernel uses for out-of-range taps).
#include "bench.hpp"
#include "leetcode_746_min_cost_climbing_stairs.h"

// c[i] = cost (>= 0, integer so min-plus path sums stay exact).
static void build_inputs(int32_t* c, size_t N, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> cost(0, 100);
  for (size_t i = 0; i < N; ++i) c[i] = (int32_t)cost(rng);
}

static void serial(const int32_t* c, int32_t* out, int num_rows, int row_size) {
  const size_t N = (size_t)num_rows * (size_t)row_size;
  for (size_t i = 0; i < N; ++i) {
    int32_t f;
    if (i == 0) f = 0;                                                 // dp(0)=0: free start on step 0
    else if (i == 1) f = std::min(0, c[0] + out[0]);                   // dp(1)=0: free start on step 1
    else f = std::min(c[i - 1] + out[i - 1], c[i - 2] + out[i - 2]);   // reach i by leaving i-1 or i-2
    out[i] = f;
  }
}

int main(int argc, char** argv) {
  using K = leetcode_746_min_cost_climbing_stairs;
  using Kb = leetcode_746_min_cost_climbing_stairs_baseline;
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/1, /*cols=*/1 << 22);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<int32_t> c(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n);
  build_inputs(c.data(), d.n, cfg.seed);
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(c.data(), d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(c.data(), d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(c.data(), out_serial.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs == 0.0) && (eb.max_abs == 0.0);  // int32: exact integer match
  bench::report3("leetcode-746-min-cost-climbing-stairs", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
