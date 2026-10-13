// Best Time to Buy and Sell Stock IV (LeetCode 188): with at most k transactions and daily
// prices g(y), maximize profit. Two-state mutual recurrence over x = #transactions, y = day,
// in TropMax (max-plus):
//   f_1(x,y) = max(f_1(x,y-1),        f_2(x-1,y-1) - g(y))   # best profit holding a share
//   f_2(x,y) = max(f_1(x,y-1) + g(y), f_2(x,y-1))            # best profit not holding
// Seeded f_2(0,0)=0 (zero profit, zero transactions); row 0 then stays f_2==0 / f_1==-inf (the
// "0 transactions" base), and rows x>=1 accumulate the k-transaction profit. The kernel emits g(y)
// via a column-indexed ComputedSignal2D reading an external price buffer registered out-of-band by
// stock_set_prices() (keeps the constructor `(int row_size)` schedule-independent -- see
// specs/registry.py). The serial loop fills the same buffer the same way and evaluates the
// identical recurrence, so int32 max-plus makes it an exact bit-for-bit comparison.
#include "bench.hpp"
#include "leetcode_188_best_time_stock_iv.h"

static constexpr int32_t NEGINF = INT32_MIN / 8;  // TropMax zero (-inf) sentinel, matches common.h

// Day prices g(y) in [1, 100] from a position hash; MUST match STOCK_PRICE_SETUP in the registry
// so opt/baseline/serial all agree. Prices are column-indexed (same for every transaction row).
static void build_prices(int32_t* g, size_t n) {
  for (size_t j = 0; j < n; ++j) {
    unsigned h = (unsigned)j * 2654435761u + 1013904223u; h ^= h >> 15;
    g[j] = (int32_t)(1 + (int)(h % 100u));
  }
}

// `f1` and `f2` are caller-owned rows*cols scratch buffers (allocated outside the timed region).
static void serial(const int32_t* g, int32_t* out, int32_t* f1, int32_t* f2, int rows, int cols) {
  for (int x = 0; x < rows; ++x) {
    for (int y = 0; y < cols; ++y) {
      size_t idx = (size_t)x * cols + y;
      int32_t f1_prev = (y >= 1) ? f1[idx - 1] : NEGINF;                                     // f_1(x, y-1)
      int32_t f2_diag = (x >= 1 && y >= 1) ? f2[(size_t)(x - 1) * cols + (y - 1)] : NEGINF;   // f_2(x-1, y-1)
      f1[idx] = std::max(f1_prev, f2_diag - g[y]);                 // f_1(x,y) = max(f_1(x,y-1), f_2(x-1,y-1)-g)
      int32_t sell = f1_prev + g[y];                               // f_1(x,y-1) + g(y)
      int32_t f2_prev = (y >= 1) ? f2[idx - 1] : NEGINF;           // f_2(x, y-1)
      int32_t seed = (x == 0 && y == 0) ? 0 : NEGINF;              // seed f_2(0,0)=0 (TropMax one)
      f2[idx] = std::max(std::max(sell, f2_prev), seed);           // f_2(x,y)
      out[idx] = f2[idx];
    }
  }
}

int main(int argc, char** argv) {
  using K = leetcode_188_best_time_stock_iv;
  using Kb = leetcode_188_best_time_stock_iv_baseline;
  bench::Config cfg = bench::parse(argc, argv, /*rows=*/2048, /*cols=*/2048);
  bench::Dims<K> d = bench::physical_dims2<K, Kb>(cfg.num_rows, cfg.row_size);
  std::vector<int32_t> g((size_t)d.cols + 64);  // day prices (column-indexed), + halo for lane over-read
  build_prices(g.data(), g.size());
  stock_set_prices(g.data());  // register out-of-band; each kernel caches this pointer at construction
  std::vector<int32_t> out_opt(d.n), out_base(d.n), out_serial(d.n);
  std::vector<int32_t> f1(d.n), f2(d.n);  // serial scratch
  bench::Timing topt = bench::run_vecrec(
      [&] { return K(d.cols); }, d.rows, d.cols, out_opt.data(), cfg.warmup, cfg.iters);
  bench::Timing tbase = bench::run_vecrec(
      [&] { return Kb(d.cols); }, d.rows, d.cols, out_base.data(), cfg.warmup, cfg.iters);
  bench::Timing ts = bench::run_serial(
      [&] { serial(g.data(), out_serial.data(), f1.data(), f2.data(), d.rows, d.cols); }, cfg.warmup, cfg.iters);
  bench::ErrStats eb = bench::compute_err(out_base.data(), out_serial.data(), d.n);
  bench::ErrStats eo = bench::compute_err(out_opt.data(), out_serial.data(), d.n);
  bool ok = (eo.max_abs == 0.0) && (eb.max_abs == 0.0);  // int32 max-plus: exact match
  bench::report3("leetcode-188-best-time-stock-iv", d.n, ts, tbase, topt, eb, eo, ok);
  return ok ? 0 : 1;
}
