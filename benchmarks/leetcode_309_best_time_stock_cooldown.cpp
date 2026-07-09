// Best Time to Buy and Sell Stock with Cooldown (LeetCode 309): unlimited transactions, but a
// one-day cooldown after every sale. The classic 3-state DP (rest/hold/sold) collapses -- by
// inlining sold[i-1] = hold[i-2] + g[i-1] -- to a 2-state mutual (max,+) recurrence:
//   rest[i] = max(rest[i-1], hold[i-2] + g[i-1])   # idle: post-cooldown / never held
//   hold[i] = max(hold[i-1], rest[i-1] - g[i])     # holding a share (buy reads rest = cooldown)
// Prices +g(i-1) and -g(i) ride on the cross taps as time-varying coefficients. Two seeds carry
// the base case: rest[0]=0 (zero profit) and hold[0]=-g[0] (buying on day 0). Output is rest, the
// running max profit while not holding. Integer prices keep the max-plus arithmetic exact, so this
// loop -- which evaluates the same two scans in the same order -- matches the kernel bit-for-bit.
#include "bench.hpp"
#include "leetcode_309_best_time_stock_cooldown.h"

static constexpr int32_t NEGINF = INT32_MIN / 8;  // TropMax zero (-inf) sentinel, matches common.h

static void build_prices(int32_t* g, size_t n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> price(1, 100);
  for (size_t i = 0; i < n; ++i) g[i] = (int32_t)price(rng);
}

static void serial(const int32_t* g, int32_t* out, int num_rows, int row_size) {
  const size_t N = (size_t)num_rows * (size_t)row_size;
  std::vector<int32_t> hold(N);
  for (int r = 0; r < num_rows; ++r) {
    for (int i = 0; i < row_size; ++i) {
      size_t idx = (size_t)r * row_size + i;
      int32_t rest_seed = (i == 0) ? 0 : NEGINF;          // rest[0] = 0 (base rest[-1] = 0)
      int32_t hold_seed = (i == 0) ? -g[idx] : NEGINF;    // hold[0] = -g[0] (buy on day 0)
      int32_t rest1 = (i >= 1) ? out[idx - 1] : NEGINF;   // rest[i-1]
      int32_t hold1 = (i >= 1) ? hold[idx - 1] : NEGINF;  // hold[i-1]
      int32_t hold2_g = (i >= 2) ? hold[idx - 2] + g[idx - 1] : NEGINF;  // hold[i-2] + g(i-1)
      int32_t rest_i = std::max(rest_seed, std::max(rest1, hold2_g));
      int32_t hold_i = std::max(hold_seed, std::max(hold1, rest1 - g[idx]));
      out[idx] = rest_i;
      hold[idx] = hold_i;
    }
  }
}

int main(int argc, char** argv) {
  using K = leetcode_309_best_time_stock_cooldown;
  using Kb = leetcode_309_best_time_stock_cooldown_baseline;
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/1, /*cols=*/1 << 22);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<int32_t> g(d.n), out_opt(d.n), out_base(d.n), out_serial(d.n);
  build_prices(g.data(), d.n, cfg.seed);
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(g.data(), d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(g.data(), d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(g.data(), out_serial.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs == 0.0) && (eb.max_abs == 0.0);  // int32 max-plus: exact match
  bench::report3("leetcode-309-best-time-stock-cooldown", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
