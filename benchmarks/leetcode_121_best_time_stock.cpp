// Best Time to Buy and Sell Stock (LeetCode 121): one transaction, the classic O(n) scan
//   f1(i) = min(f1(i-1), g[i])            # cheapest price so far
//   f2(i) = max(f2(i-1), g[i] - f1(i-1))  # best profit selling on day i (buy strictly earlier)
// f1 is a min-scan and f2 a max-scan. The kernel keeps both in one TropMax (max,+) semiring by
// carrying pm(i) = -f1(i) = max(pm(i-1), -g[i]); then g[i] - f1(i-1) = g[i] + pm(i-1). Output is
// f2 (the max profit). Integer prices keep the max-plus arithmetic exact, so this loop -- which
// evaluates the same two prefix scans in the same order -- matches the kernel bit-for-bit.
#include "bench.hpp"
#include "leetcode_121_best_time_stock.h"

static constexpr int32_t NEGINF = INT32_MIN / 8;  // TropMax zero (-inf) sentinel, matches common.h

static void build_prices(int32_t* g, size_t n, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> price(1, 100);
  for (size_t i = 0; i < n; ++i) g[i] = (int32_t)price(rng);
}

static void serial(const int32_t* g, int32_t* out, int num_rows, int row_size) {
  for (int r = 0; r < num_rows; ++r) {
    int32_t pm = NEGINF, f2 = NEGINF;  // pm(-1) = -inf (= -min over empty), f2(-1) = -inf; reset per row
    for (int i = 0; i < row_size; ++i) {
      size_t idx = (size_t)r * row_size + i;
      int32_t pm_prev = pm;                         // pm(i-1) = -min g[0..i-1]
      int32_t buy_low = g[idx] + pm_prev;           // g(i) - min g[0..i-1]
      f2 = std::max(f2, buy_low);                   // f2(i) = max(f2(i-1), g(i)-f1(i-1))
      pm = std::max(pm_prev, -g[idx]);              // pm(i) = max(pm(i-1), -g(i))
      out[idx] = f2;
    }
  }
}

int main(int argc, char** argv) {
  using K = leetcode_121_best_time_stock;
  using Kb = leetcode_121_best_time_stock_baseline;
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
  bench::report3("leetcode-121-best-time-stock", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
